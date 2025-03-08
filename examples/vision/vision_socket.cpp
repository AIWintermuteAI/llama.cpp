#include "llama.h"
#include "common.h"
#include "arg.h"
#include "log.h"
#include "sampling.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include "../../common/json.hpp"
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using json = nlohmann::json;

typedef struct {
    bool initialized;
    llama_context* ctx;
    const llama_model* model;
    const llama_vocab* vocab;
    struct common_sampler* smpl;
} server_state_t;

static server_state_t state = { false, nullptr, nullptr, nullptr, nullptr };

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [--image img_path] [-p prompt]\n", argv[0]);
    printf("\n");
}

static llama_vision_bitmap * load_image_from_file(const char * fname) {
    std::ifstream file(fname, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Unable to open file");
    }
    std::vector<char> image_bytes = std::vector<char>(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>());
    // decode image to byte array
    int nx, ny, nc;
    auto * bytes = (unsigned char *) image_bytes.data();
    auto * img = stbi_load_from_memory(bytes, image_bytes.size(), &nx, &ny, &nc, 3);
    if (!img) {
        throw std::runtime_error("failed to decode image bytes");
    }
    // printf("nx=%d ny=%d nc=%d\n", nx, ny, nc);
    // GGML_ASSERT(nc == 3);
    // for (int y = 0; y < ny; y++) {
    //     for (int x = 0; x < nx; x++) {
    //         unsigned char * pix = img + x*nc + y*nc*nx;
    //         printf("%02x%02x%02x ", pix[0], pix[1], pix[2]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    llama_vision_bitmap * result = llama_vision_bitmap_init(nx, ny);
    memcpy(result->data, img, nx*ny*3);
    stbi_image_free(img);
    return result;
}

// split string by a `std::string delim` instead of `char delim`
static std::vector<std::string> string_split_str(std::string s, const std::string & delimiter) {
    std::vector<std::string> tokens;
    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        tokens.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    tokens.push_back(s);
    return tokens;
}

struct tokenized_part {
    llama_tokens tokens;
    bool is_image;
};

// TODO: this function is hacky, need to be improved
// static const llama_token TOKEN_IMG_PLACEMENT = -1000;
static const std::string IMG_PLACEMENT = "<img_placement>";
static std::vector<tokenized_part> tokenize_with_img_placement(
        const llama_vocab * vocab,
        const std::string & text,
        bool   add_special,
        bool   parse_special) {
    std::vector<std::string> parts = string_split_str(text, IMG_PLACEMENT);
    std::vector<tokenized_part> output;
    for (const auto & part : parts) {
        //printf("tokenizing part: %s\n", part.c_str());
        bool add_bos = &parts.front() == &part;
        auto tokens = common_tokenize(vocab, part, add_special && add_bos, parse_special);
        if (tokens.empty()) {
            continue;
        }
        output.push_back({std::move(tokens), false});
        if (&parts.back() != &part) {
            // add image token to middle of 2 parts
            output.push_back({{}, true});
        }
    }
    return output;
}

static void send_error_response(int id, const std::string& error_msg, char* resp_buffer, size_t resp_buffer_size) {
    json err = {
        {"id", id},
        {"success", false},
        {"error", error_msg}
    };
    snprintf(resp_buffer, resp_buffer_size, "%s\n", err.dump().c_str());
}

static void send_success_response(int id, const std::string& generated_text, char* resp_buffer, size_t resp_buffer_size) {
    json resp = {
        {"id", id},
        {"success", true},
        {"result", {
            {"text", generated_text}
        }}
    };
    snprintf(resp_buffer, resp_buffer_size, "%s\n", resp.dump().c_str());
}

static void handle_clear_kv_cache_request(const json& request, char* resp_buffer, size_t resp_buffer_size) {
    if (!state.initialized) {
        send_error_response(request["id"], "Server not initialized", resp_buffer, resp_buffer_size);
        return;
    }
    llama_kv_cache_clear(state.ctx);
}

static void handle_inference_request(const json& request, char* resp_buffer, size_t resp_buffer_size) {
    if (!state.initialized) {
        send_error_response(request["id"], "Server not initialized", resp_buffer, resp_buffer_size);
        return;
    }

    if (!request.contains("infer")) {
        send_error_response(request["id"], "Missing 'infer' object in request", resp_buffer, resp_buffer_size);
        return;
    }

    const json& infer = request["infer"];
    if (!infer.contains("image_path")) {
        send_error_response(request["id"], "Missing 'image_path' in infer request", resp_buffer, resp_buffer_size);
        return;
    }
    if (!infer.contains("prompt")) {
        send_error_response(request["id"], "Missing 'prompt' in infer request", resp_buffer, resp_buffer_size);
        return;
    }

    std::string image_path = infer["image_path"];
    std::string prompt = infer["prompt"];
    int n_predict = infer.value("n_predict", 64);

    // Process image
    llama_vision_tokens* img_tokens = nullptr;
    try {
        llama_vision_bitmap* img = load_image_from_file(image_path.c_str());
        LOG_INF("loaded image %s, size = %d x %d\n", image_path.c_str(), img->nx, img->ny);
        img_tokens = llama_vision_tokenize(state.ctx, img);
        if (!img_tokens) {
            send_error_response(request["id"], "Failed to create image tokens", resp_buffer, resp_buffer_size);
            return;
        }
        if (llama_vision_encode(state.ctx, img_tokens)) {
            send_error_response(request["id"], "Failed to encode image", resp_buffer, resp_buffer_size);
            return;
        }
        LOG_INF("encoded image\n");
    } catch (const std::exception& e) {
        send_error_response(request["id"], std::string("Image processing error: ") + e.what(), resp_buffer, resp_buffer_size);
        return;
    }

    llama_batch batch = llama_batch_init(llama_n_batch(state.ctx), 0, 1);
    int n_past = 0;
    int n_prompt = 0;
    std::stringstream output;

    // Process prompt
    std::vector<tokenized_part> parts = tokenize_with_img_placement(state.vocab, prompt, true, true);
    for (const tokenized_part& part : parts) {
        if (!part.is_image) {
            for (const llama_token& token : part.tokens) {
                common_batch_add(batch, token, n_past++, {0}, &part == &parts.back());
            }
            LOG_INF("eval text batch (%d tokens)\n", batch.n_tokens);
            if (llama_decode(state.ctx, batch)) {
                send_error_response(request["id"], "Failed to decode text prompt", resp_buffer, resp_buffer_size);
                return;
            }
        } else {
            auto* img_embd = llama_vision_get_output_tensor(state.ctx);
            llama_batch batch_img = llama_batch_get_one_from_tensor(img_embd, n_past, 0);
            n_past += batch_img.n_tokens;
            LOG_INF("eval image batch (%d embeddings)\n", batch_img.n_tokens);
            if (llama_decode(state.ctx, batch_img)) {
                send_error_response(request["id"], "Failed to decode image prompt", resp_buffer, resp_buffer_size);
                return;
            }
            llama_batch_free(batch_img);
        }
    }
    n_prompt = n_past;
    LOG_INF("prompt processed, %d tokens\n", n_prompt);

    // Generate response
    while (true) {
        int n_generated = n_past - n_prompt;
        if (n_generated > n_predict) {
            break;
        }

        llama_token token_id = common_sampler_sample(state.smpl, state.ctx, -1);
        common_sampler_accept(state.smpl, token_id, true);
        output << common_token_to_piece(state.ctx, token_id);

        if (llama_vocab_is_eog(state.vocab, token_id)) {
            break;
        }

        // Eval the token
        common_batch_clear(batch);
        common_batch_add(batch, token_id, n_past++, {0}, true);
        if (llama_decode(state.ctx, batch)) {
            send_error_response(request["id"], "Failed to decode token", resp_buffer, resp_buffer_size);
            return;
        }
    }

    send_success_response(request["id"], output.str(), resp_buffer, resp_buffer_size);
}

static void handle_init_request(const json& request, char* resp_buffer, size_t resp_buffer_size) {
    // if (state.initialized) {
    //     send_error_response(request["id"], "Server already initialized", resp_buffer, resp_buffer_size);
    //     return;
    // }

    if (!request.contains("init")) {
        send_error_response(request["id"], "Missing 'init' object in request", resp_buffer, resp_buffer_size);
        return;
    }

    const json& init = request["init"];
    if (!init.contains("model_path")) {
        send_error_response(request["id"], "Missing 'model_path' in init request", resp_buffer, resp_buffer_size);
        return;
    }

    json resp = {
        {"id", request["id"]},
        {"success", true},
        {"message", "Server initialized successfully"}
    };
    snprintf(resp_buffer, resp_buffer_size, "%s\n", resp.dump().c_str());
}

void handle_json_message(const json& msg, char* resp_buffer, size_t resp_buffer_size) {
    if (!msg.contains("id")) {
        send_error_response(-1, "Missing 'id' field in message", resp_buffer, resp_buffer_size);
        return;
    }

    if (msg.contains("init")) {
        handle_init_request(msg, resp_buffer, resp_buffer_size);
    }
    else if (msg.contains("infer")) {
        handle_inference_request(msg, resp_buffer, resp_buffer_size);
    }
    else if (msg.contains("clear_kv_cache")) {
        handle_clear_kv_cache_request(msg, resp_buffer, resp_buffer_size);
    }
    else {
        send_error_response(msg["id"], "Unknown request type", resp_buffer, resp_buffer_size);
    }
}

int socket_main(const char* socket_path) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        LOG_ERR("Failed to create socket\n");
        return 1;
    }

    struct sockaddr_un addr = { 0 };
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);
    unlink(socket_path);

    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        LOG_ERR("Failed to bind to socket\n");
        return 1;
    }

    if (listen(fd, 5) < 0) {
        LOG_ERR("Failed to listen on socket\n");
        return 1;
    }

    LOG_INF("Server listening on %s\n", socket_path);

    char buffer[65536];
    char resp_buffer[65536];

    while (true) {
        int connfd = accept(fd, nullptr, nullptr);
        if (connfd < 0) {
            LOG_ERR("Failed to accept connection\n");
            continue;
        }

        LOG_INF("New connection accepted\n");

        while (true) {
            ssize_t n = read(connfd, buffer, sizeof(buffer) - 1);
            if (n <= 0) {
                break;
            }
            buffer[n] = '\0';

            try {
                json msg = json::parse(buffer);
                handle_json_message(msg, resp_buffer, sizeof(resp_buffer));
                write(connfd, resp_buffer, strlen(resp_buffer));
            }
            catch (const json::exception& e) {
                send_error_response(-1, std::string("JSON parsing error: ") + e.what(), resp_buffer, sizeof(resp_buffer));
                write(connfd, resp_buffer, strlen(resp_buffer));
            }
        }

        close(connfd);
        LOG_INF("Connection closed\n");
    }

    return 0;
}

int main(int argc, char** argv) {
    // if (strcmp(argv[1], "--socket") != 0) {
    //     printf("Usage: %s --socket <socket_path>\n", argv[0]);
    //     return 1;
    // }

    common_params params;
    params.n_predict = -1;
    params.n_batch = 2048;
    params.n_ubatch = 1024;
    params.n_gpu_layers = 99;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_VISION, print_usage)) {
        return 1;
    }

    common_init();
    common_init_result llama_init = common_init_from_params(params);
    state.ctx = llama_init.context.get();
    state.model = llama_init.model.get();
    state.vocab = llama_model_get_vocab(state.model);

    if (!state.model) {
        return 1;
    }

    state.smpl = common_sampler_init(state.model, params.sampling);
    state.initialized = true;

    return socket_main("/tmp/vision.sock");
}
