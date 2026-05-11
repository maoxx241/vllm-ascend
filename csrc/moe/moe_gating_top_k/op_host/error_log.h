#ifndef OPS_BUILT_IN_OP_TILING_ERROR_LOG_H_
#define OPS_BUILT_IN_OP_TILING_ERROR_LOG_H_

#include <string>
#include "toolchain/slog.h"

namespace optiling {

inline const char *GetLogOpName(const char *opname)
{
    return opname == nullptr ? "MoeGatingTopK" : opname;
}

inline const char *GetLogOpName(const std::string &opname)
{
    return opname.empty() ? "MoeGatingTopK" : opname.c_str();
}

template <typename Context>
inline const char *GetLogOpName(const Context *context)
{
    if (context == nullptr || context->GetNodeName() == nullptr) {
        return "MoeGatingTopK";
    }
    return context->GetNodeName();
}

}  // namespace optiling

#define OP_LOGI(opname, ...)
#define OP_LOGW(opname, ...)             \
    do {                                 \
        printf("[WARN][%s] ", optiling::GetLogOpName(opname)); \
        printf(__VA_ARGS__);             \
        printf("\n");                    \
    } while (0)

#define OP_LOGE_WITHOUT_REPORT(opname, ...) \
    do {                                    \
        printf("[ERRORx][%s] ", optiling::GetLogOpName(opname));  \
        printf(__VA_ARGS__);                \
        printf("\n");                       \
    } while (0)

#define OP_LOGE(opname, ...)              \
    do {                                  \
        printf("[ERROR][%s] ", optiling::GetLogOpName(opname));    \
        printf(__VA_ARGS__);              \
        printf("\n");                     \
    } while (0)

#define OP_LOGD(opname, ...)

namespace optiling {

#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...)   \
    do {                                                         \
        OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__); \
    } while (0)


#define OP_CHECK_IF(cond, log_func, expr) \
    do {                                      \
        if (cond) {                           \
            log_func;                         \
            expr;                             \
        }                                     \
    } while (0)



#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr)                          \
    do {                                                                  \
        if ((ptr) == nullptr) {                                           \
            OP_LOGE(context->GetNodeType(), "%s is null", #ptr);               \
            return ge::GRAPH_FAILED;                                      \
        }                                                                 \
    } while (0)

}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_ERROR_LOG_H_
