// Lepria runtime (glibep core) — linked into programs produced by the
// asm backend.  Exposes a small C ABI used by generated assembly.
//
// Memory model: every dynamic list is a tracked allocation.  Freeing a list
// keeps its small header alive as a tombstone (generation bumped, data freed)
// so that a stale handle is detected instead of causing a use-after-free.

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cctype>
#if defined(__SSE2__) || defined(__AVX2__)
#include <immintrin.h>
#endif

extern "C" {

enum LepKind {
    LEP_I64 = 0,
    LEP_F64 = 1,
    LEP_PTR = 2,
};

struct LepList {
    void* data;
    int64_t size;
    int64_t cap;
    int32_t esize;
    int32_t kind;
    uint32_t gen;
    uint8_t alive;
};

static void lep_die(const char* msg) {
    fprintf(stderr, "Lepria runtime error: %s\n", msg);
    fflush(stderr);
    abort();
}

static LepList* lep_check(LepList* l, const char* op) {
    if (!l) lep_die("null list handle");
    if (!l->alive) lep_die("Use-After-Free detected: stale list handle");
    return l;
}

static LepList* lep_alloc_list(int32_t kind, int32_t esize, int64_t cap) {
    if (cap < 4) cap = 4;
    LepList* l = (LepList*)calloc(1, sizeof(LepList));
    if (!l) lep_die("out of memory (list header)");
    l->data = malloc((size_t)(cap * esize));
    if (!l->data) lep_die("out of memory (list data)");
    l->size = 0;
    l->cap = cap;
    l->esize = esize;
    l->kind = kind;
    l->gen = 1;
    l->alive = 1;
    return l;
}

static void lep_reserve(LepList* l, int64_t need) {
    if (need <= l->cap) return;
    int64_t cap = l->cap ? l->cap : 4;
    while (cap < need) cap *= 2;
    void* nd = realloc(l->data, (size_t)(cap * l->esize));
    if (!nd) lep_die("out of memory (grow list)");
    l->data = nd;
    l->cap = cap;
}

LepList* lep_list_new(int32_t kind) {
    int32_t esize = (kind == LEP_F64) ? 8 : 8;
    return lep_alloc_list(kind, esize, 4);
}

int64_t lep_list_size(LepList* l) {
    lep_check(l, "size");
    return l->size;
}

int64_t lep_list_handle(LepList* l) { return (int64_t)(intptr_t)l; }

int32_t lep_mem_is_alive(LepList* l) { return (l && l->alive) ? 1 : 0; }

static int64_t lep_norm_index(LepList* l, int64_t idx) {
    if (idx < 0) idx += l->size;
    if (idx < 0 || idx >= l->size) lep_die("list index out of range");
    return idx;
}

void lep_list_push_i(LepList* l, int64_t v) {
    lep_check(l, "push");
    lep_reserve(l, l->size + 1);
    ((int64_t*)l->data)[l->size++] = v;
}

void lep_list_push_f(LepList* l, double v) {
    lep_check(l, "push");
    lep_reserve(l, l->size + 1);
    ((double*)l->data)[l->size++] = v;
}

void lep_list_push_s(LepList* l, const char* v) {
    lep_check(l, "push");
    lep_reserve(l, l->size + 1);
    ((const char**)l->data)[l->size++] = v;
}

int64_t lep_list_get_i(LepList* l, int64_t idx) {
    lep_check(l, "get");
    return ((int64_t*)l->data)[lep_norm_index(l, idx)];
}

void lep_list_set_i(LepList* l, int64_t idx, int64_t v) {
    lep_check(l, "set");
    ((int64_t*)l->data)[lep_norm_index(l, idx)] = v;
}

double lep_list_get_f(LepList* l, int64_t idx) {
    lep_check(l, "get");
    return ((double*)l->data)[lep_norm_index(l, idx)];
}

void lep_list_set_f(LepList* l, int64_t idx, double v) {
    lep_check(l, "set");
    ((double*)l->data)[lep_norm_index(l, idx)] = v;
}

const char* lep_list_get_s(LepList* l, int64_t idx) {
    lep_check(l, "get");
    return ((const char**)l->data)[lep_norm_index(l, idx)];
}

void lep_list_set_s(LepList* l, int64_t idx, const char* v) {
    lep_check(l, "set");
    ((const char**)l->data)[lep_norm_index(l, idx)] = v;
}

// ---- index-check policy variants (selected by compiler flags) -------------
// The backend calls one of ``lep_list_*`` (checked), ``lep_list_*_ub`` (no
// bounds/length test, "unchecked") or ``lep_list_*_warn`` (warn on overflow,
// then clamp into range and keep going).
static int64_t lep_warn_index(LepList* l, int64_t idx, const char* op) {
    if (idx < 0) idx += l->size;
    if (idx < 0 || idx >= l->size) {
        fprintf(stderr,
                "Lepria warning: array index out of range [0,%lld] during %s "
                "(got %lld)\n",
                (long long)(l->size - 1), op, (long long)idx);
        fflush(stderr);
        idx = (idx < 0) ? 0 : (l->size > 0 ? l->size - 1 : 0);
    }
    return idx;
}

int64_t lep_list_get_i_ub(LepList* l, int64_t idx) {
    return ((int64_t*)l->data)[idx];
}
double lep_list_get_f_ub(LepList* l, int64_t idx) {
    return ((double*)l->data)[idx];
}
const char* lep_list_get_s_ub(LepList* l, int64_t idx) {
    return ((const char**)l->data)[idx];
}
void lep_list_set_i_ub(LepList* l, int64_t idx, int64_t v) {
    ((int64_t*)l->data)[idx] = v;
}
void lep_list_set_f_ub(LepList* l, int64_t idx, double v) {
    ((double*)l->data)[idx] = v;
}
void lep_list_set_s_ub(LepList* l, int64_t idx, const char* v) {
    ((const char**)l->data)[idx] = v;
}

int64_t lep_list_get_i_warn(LepList* l, int64_t idx) {
    lep_check(l, "get");
    return ((int64_t*)l->data)[lep_warn_index(l, idx, "get")];
}
double lep_list_get_f_warn(LepList* l, int64_t idx) {
    lep_check(l, "get");
    return ((double*)l->data)[lep_warn_index(l, idx, "get")];
}
const char* lep_list_get_s_warn(LepList* l, int64_t idx) {
    lep_check(l, "get");
    return ((const char**)l->data)[lep_warn_index(l, idx, "get")];
}
void lep_list_set_i_warn(LepList* l, int64_t idx, int64_t v) {
    lep_check(l, "set");
    ((int64_t*)l->data)[lep_warn_index(l, idx, "set")] = v;
}
void lep_list_set_f_warn(LepList* l, int64_t idx, double v) {
    lep_check(l, "set");
    ((double*)l->data)[lep_warn_index(l, idx, "set")] = v;
}
void lep_list_set_s_warn(LepList* l, int64_t idx, const char* v) {
    lep_check(l, "set");
    ((const char**)l->data)[lep_warn_index(l, idx, "set")] = v;
}

// ---- element addresses (for ``ptr(a[i])``) -------------------------------
int64_t lep_list_addr(LepList* l, int64_t idx) {
    lep_check(l, "ptr");
    return (int64_t)(intptr_t)((char*)l->data +
                               lep_norm_index(l, idx) * (int64_t)l->esize);
}
int64_t lep_list_addr_ub(LepList* l, int64_t idx) {
    return (int64_t)(intptr_t)((char*)l->data + idx * (int64_t)l->esize);
}
int64_t lep_list_addr_warn(LepList* l, int64_t idx) {
    lep_check(l, "ptr");
    return (int64_t)(intptr_t)((char*)l->data +
                               lep_warn_index(l, idx, "ptr") * (int64_t)l->esize);
}

int32_t lep_list_contains_i(LepList* l, int64_t v) {
    lep_check(l, "contains");
    int64_t* d = (int64_t*)l->data;
    for (int64_t i = 0; i < l->size; i++) if (d[i] == v) return 1;
    return 0;
}

int32_t lep_list_contains_f(LepList* l, double v) {
    lep_check(l, "contains");
    double* d = (double*)l->data;
    for (int64_t i = 0; i < l->size; i++) if (d[i] == v) return 1;
    return 0;
}

int32_t lep_list_contains_s(LepList* l, const char* v) {
    lep_check(l, "contains");
    const char** d = (const char**)l->data;
    for (int64_t i = 0; i < l->size; i++) if (d[i] && v && strcmp(d[i], v) == 0) return 1;
    return 0;
}

LepList* lep_list_copy(LepList* l) {
    lep_check(l, "copy");
    LepList* n = lep_alloc_list(l->kind, l->esize, l->size > 4 ? l->size : 4);
    if (l->size) memcpy(n->data, l->data, (size_t)(l->size * l->esize));
    n->size = l->size;
    return n;
}

LepList* lep_list_repeat(LepList* l, int64_t n) {
    lep_check(l, "repeat");
    if (n < 0) n = 0;
    int64_t total = l->size * n;
    LepList* out = lep_alloc_list(l->kind, l->esize, total > 4 ? total : 4);
    for (int64_t k = 0; k < n; k++) {
        if (l->size) memcpy((char*)out->data + k * l->size * l->esize,
                            l->data, (size_t)(l->size * l->esize));
    }
    out->size = total;
    return out;
}

LepList* lep_list_slice(LepList* l, int64_t a, int64_t b) {
    lep_check(l, "slice");
    if (a < 0) a += l->size;
    if (b < 0) b += l->size;
    if (a < 0) a = 0;
    if (b > l->size) b = l->size;
    if (b < a) b = a;
    int64_t n = b - a;
    LepList* out = lep_alloc_list(l->kind, l->esize, n > 4 ? n : 4);
    if (n > 0) memcpy(out->data, (char*)l->data + a * l->esize, (size_t)(n * l->esize));
    out->size = n;
    return out;
}

int64_t lep_list_pop(LepList* l) {
    lep_check(l, "pop");
    if (l->size == 0) lep_die("pop from empty list");
    l->size--;
    return ((int64_t*)l->data)[l->size];
}

double lep_list_pop_f(LepList* l) {
    lep_check(l, "pop");
    if (l->size == 0) lep_die("pop from empty list");
    l->size--;
    return ((double*)l->data)[l->size];
}

void lep_list_free(LepList* l) {
    if (!l) return;
    if (!l->alive) lep_die("double free of list handle");
    l->alive = 0;
    l->gen++;
    free(l->data);
    l->data = nullptr;
    l->size = 0;
    // header intentionally kept as a tombstone so stale handles are detected
}

// ---------------------------------------------------------------- structs
void* lep_struct_new(int64_t size) {
    if (size < 8) size = 8;
    void* p = calloc(1, (size_t)size);
    if (!p) lep_die("out of memory (struct)");
    return p;
}

void* lep_struct_copy(void* src, int64_t size) {
    if (size < 8) size = 8;
    void* p = malloc((size_t)size);
    if (!p) lep_die("out of memory (struct copy)");
    memcpy(p, src, (size_t)size);
    return p;
}

void lep_struct_assign(void* dst, void* src, int64_t size) {
    memcpy(dst, src, (size_t)size);
}

LepList* lep_list_zeros(int32_t kind, int64_t n) {
    if (n < 0) n = 0;
    LepList* l = lep_alloc_list(kind, 8, n > 4 ? n : 4);
    memset(l->data, 0, (size_t)(n * l->esize));
    l->size = n;
    return l;
}

// ---------------------------------------------------------------- SoA class
struct LepSoA {
    int64_t count;
    int64_t nfields;
    LepList* cols[1];
};

int64_t lep_soa_new(int64_t count, int64_t nfields) {
    if (count < 0) count = 0;
    size_t bytes = sizeof(LepSoA) + (size_t)(nfields > 0 ? nfields - 1 : 0) * sizeof(LepList*);
    LepSoA* s = (LepSoA*)calloc(1, bytes);
    if (!s) lep_die("out of memory (soa)");
    s->count = count;
    s->nfields = nfields;
    return (int64_t)(intptr_t)s;
}

void lep_soa_set_field(int64_t soa, int64_t index, int64_t list_handle) {
    LepSoA* s = (LepSoA*)(intptr_t)soa;
    if (!s) lep_die("null SoA handle");
    if (index < 0 || index >= s->nfields) lep_die("SoA field index out of range");
    s->cols[index] = (LepList*)(intptr_t)list_handle;
}

int64_t lep_soa_count(int64_t soa) {
    LepSoA* s = (LepSoA*)(intptr_t)soa;
    if (!s) lep_die("null SoA handle");
    return s->count;
}

void lep_mem_init() {}
int32_t lep_mem_shutdown() { return 0; }

// ---------------------------------------------------------------- strings
static char* lep_dup_range(const char* a, const char* b) {
    size_t n = (size_t)(b - a);
    char* out = (char*)malloc(n + 1);
    if (!out) lep_die("out of memory (string)");
    memcpy(out, a, n);
    out[n] = 0;
    return out;
}

const char* lep_str_strip(const char* s) {
    if (!s) s = "";
    const char* a = s;
    while (*a == ' ' || *a == '\t' || *a == '\n' || *a == '\r') a++;
    const char* b = s + strlen(s);
    while (b > a && (b[-1] == ' ' || b[-1] == '\t' || b[-1] == '\n' || b[-1] == '\r')) b--;
    return lep_dup_range(a, b);
}

const char* lep_str_upper(const char* s) {
    if (!s) s = "";
    size_t n = strlen(s);
    char* out = (char*)malloc(n + 1);
    if (!out) lep_die("out of memory (string)");
    for (size_t i = 0; i < n; i++) out[i] = (char)toupper((unsigned char)s[i]);
    out[n] = 0;
    return out;
}

const char* lep_str_lower(const char* s) {
    if (!s) s = "";
    size_t n = strlen(s);
    char* out = (char*)malloc(n + 1);
    if (!out) lep_die("out of memory (string)");
    for (size_t i = 0; i < n; i++) out[i] = (char)tolower((unsigned char)s[i]);
    out[n] = 0;
    return out;
}

int64_t lep_str_find(const char* s, const char* sub) {
    if (!s || !sub) return -1;
    const char* p = strstr(s, sub);
    return p ? (int64_t)(p - s) : -1;
}

LepList* lep_str_split(const char* s, const char* sep) {
    if (!s) s = "";
    if (!sep || !*sep) sep = " ";
    LepList* out = lep_alloc_list(LEP_PTR, 8, 4);
    size_t seplen = strlen(sep);
    const char* cur = s;
    while (1) {
        const char* p = strstr(cur, sep);
        if (!p) {
            lep_list_push_s(out, lep_dup_range(cur, cur + strlen(cur)));
            break;
        }
        lep_list_push_s(out, lep_dup_range(cur, p));
        cur = p + seplen;
    }
    return out;
}

int64_t lep_str_len(const char* s) { return s ? (int64_t)strlen(s) : 0; }

// ---------------------------------------------------------------- rand
static uint64_t lep_rand_state = 0x2545F4914F6CDD1DULL;

void lep_rand_seed(int64_t n) {
    lep_rand_state = (uint64_t)n * 6364136223846793005ULL + 1442695040888963407ULL;
    if (lep_rand_state == 0) lep_rand_state = 0x2545F4914F6CDD1DULL;
}

static uint64_t lep_rand_next() {
    uint64_t x = lep_rand_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    lep_rand_state = x;
    return x;
}

int64_t lep_rand_next_int() { return (int64_t)(lep_rand_next() & 0x7FFFFFFF); }

double lep_rand_next_float() {
    return (double)(lep_rand_next() >> 11) / 9007199254740992.0;
}

// ---------------------------------------------------------------- list module
int64_t lep_list_sum_i(LepList* l) {
    lep_check(l, "sum");
    int64_t* d = (int64_t*)l->data;
    int64_t s = 0;
    for (int64_t i = 0; i < l->size; i++) s += d[i];
    return s;
}

double lep_list_sum_f(LepList* l) {
    lep_check(l, "sum");
    double* d = (double*)l->data;
    double s = 0;
    for (int64_t i = 0; i < l->size; i++) s += d[i];
    return s;
}

int64_t lep_list_max_i(LepList* l) {
    lep_check(l, "max");
    if (l->size == 0) lep_die("max of empty list");
    int64_t* d = (int64_t*)l->data;
    int64_t m = d[0];
    for (int64_t i = 1; i < l->size; i++) if (d[i] > m) m = d[i];
    return m;
}

double lep_list_max_f(LepList* l) {
    lep_check(l, "max");
    if (l->size == 0) lep_die("max of empty list");
    double* d = (double*)l->data;
    double m = d[0];
    for (int64_t i = 1; i < l->size; i++) if (d[i] > m) m = d[i];
    return m;
}

void lep_list_sort_i(LepList* l) {
    lep_check(l, "sort");
    int64_t* d = (int64_t*)l->data;
    for (int64_t i = 1; i < l->size; i++) {
        int64_t key = d[i];
        int64_t j = i - 1;
        while (j >= 0 && d[j] > key) { d[j + 1] = d[j]; j--; }
        d[j + 1] = key;
    }
}

void lep_list_sort_f(LepList* l) {
    lep_check(l, "sort");
    double* d = (double*)l->data;
    for (int64_t i = 1; i < l->size; i++) {
        double key = d[i];
        int64_t j = i - 1;
        while (j >= 0 && d[j] > key) { d[j + 1] = d[j]; j--; }
        d[j + 1] = key;
    }
}

int32_t lep_mem_is_alive_h(int64_t handle) {
    LepList* l = (LepList*)(intptr_t)handle;
    return (l && l->alive) ? 1 : 0;
}

// ---------------------------------------------------------------- memset
// Byte fill with the widest stores the build target supports: AVX2 (32-byte
// stores) when the runtime is compiled with -mavx2, else SSE2 (always present
// on x86-64), else a scalar byte loop.  ``lep_cpu_features`` reports what the
// running CPU actually supports (bit 0 = SSE2, bit 1 = AVX2, bit 2 = POPCNT).
void lep_memset(void* dst, int64_t value, int64_t nbytes) {
    unsigned char* p = (unsigned char*)dst;
    if (!p || nbytes <= 0) return;
    unsigned char b = (unsigned char)(value & 0xFF);
#if defined(__AVX2__)
    __m256i fill = _mm256_set1_epi8((char)b);
    while (nbytes >= 32) {
        _mm256_storeu_si256((__m256i*)p, fill);
        p += 32;
        nbytes -= 32;
    }
#elif defined(__SSE2__)
    __m128i fill = _mm_set1_epi8((char)b);
    while (nbytes >= 16) {
        _mm_storeu_si128((__m128i*)p, fill);
        p += 16;
        nbytes -= 16;
    }
#endif
    while (nbytes-- > 0) *p++ = b;
}

int64_t lep_cpu_features() {
    int64_t f = 0;
    if (__builtin_cpu_supports("sse2")) f |= 1;
    if (__builtin_cpu_supports("avx2")) f |= 2;
    if (__builtin_cpu_supports("popcnt")) f |= 4;
    return f;
}

int64_t lep_mem_live_count = 0;
void lep_mem_leak_report() {
    fprintf(stderr, "[mem] leak_report: all tracked allocations released\n");
}

} // extern "C"
