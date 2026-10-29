/* Narrow host-platform compatibility layer for the CUDA sieve. */
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include "platform.h"

#include <errno.h>
#include <limits.h>
#include <string.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <windows.h>
#else
#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#endif

double bench_monotonic_ms(void)
{
#ifdef _WIN32
    LARGE_INTEGER now, freq;
    if (!QueryPerformanceFrequency(&freq) || !QueryPerformanceCounter(&now))
        return (double)time(NULL) * 1000.0;
    return (double)now.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    /* CLOCK_MONOTONIC_RAW where there is one. CLOCK_MONOTONIC is slewed by
     * adjtimex: under WSL2, with systemd-timesyncd and Hyper-V time sync both
     * steering, it ran 5.3-7.7% fast against the card's event timer, by a
     * different amount in each 10 s, while RAW matched it to 1e-4
     * (finding 134). */
    struct timespec t;
#ifdef CLOCK_MONOTONIC_RAW
    if (!clock_gettime(CLOCK_MONOTONIC_RAW, &t))
        return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
#endif
    if (clock_gettime(CLOCK_MONOTONIC, &t))
        return (double)time(NULL) * 1000.0;
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
#endif
}

int bench_sync_stream(FILE *f)
{
#ifdef _WIN32
    return _commit(_fileno(f));
#else
    return fsync(fileno(f));
#endif
}

int64_t bench_tell(FILE *f)
{
#ifdef _WIN32
    return _ftelli64(f);
#else
    return (int64_t)ftello(f);
#endif
}

int bench_seek(FILE *f, uint64_t off)
{
    if (off > (uint64_t)INT64_MAX) { errno = EOVERFLOW; return -1; }
#ifdef _WIN32
    return _fseeki64(f, (__int64)off, SEEK_SET);
#else
    return fseeko(f, (off_t)off, SEEK_SET);
#endif
}

/* Seek to end. Separate from bench_seek because that one is SEEK_SET-only and
 * takes an unsigned offset; sizing a file needs the whence, not an offset. */
int bench_seek_end(FILE *f)
{
#ifdef _WIN32
    return _fseeki64(f, 0, SEEK_END);
#else
    return fseeko(f, 0, SEEK_END);
#endif
}

int bench_truncate(FILE *f, uint64_t off)
{
    if (off > (uint64_t)INT64_MAX) { errno = EOVERFLOW; return -1; }
#ifdef _WIN32
    {
        const errno_t e = _chsize_s(_fileno(f), (__int64)off);
        if (e) { errno = (int)e; return -1; }
        return 0;
    }
#else
    return ftruncate(fileno(f), (off_t)off);
#endif
}

int bench_fstat_stream(FILE *f, bench_stat_t *st)
{
#ifdef _WIN32
    return _fstat64(_fileno(f), st);
#else
    return fstat(fileno(f), st);
#endif
}

/* ---- durable file identity --------------------------------------------- */

#ifdef _WIN32
/* One place that turns a Win32 handle into an id, so the stream and path
 * forms cannot drift. FILE_SHARE_DELETE matters: the candidate staging file
 * is open in this process while we ask, and without it the open below fails
 * with a sharing violation on the very file we are asking about. */
static int bench_file_id_from_handle(HANDLE h, bench_file_id_t *id)
{
    BY_HANDLE_FILE_INFORMATION bhfi;
    if (h == INVALID_HANDLE_VALUE || !GetFileInformationByHandle(h, &bhfi)) {
        errno = EINVAL;
        return -1;
    }
    id->volume = (uint64_t)bhfi.dwVolumeSerialNumber;
    id->index  = ((uint64_t)bhfi.nFileIndexHigh << 32) |
                 (uint64_t)bhfi.nFileIndexLow;
    return 0;
}
#endif

int bench_file_id_stream(FILE *f, bench_file_id_t *id)
{
#ifdef _WIN32
    return bench_file_id_from_handle((HANDLE)_get_osfhandle(_fileno(f)), id);
#else
    struct stat st;
    if (fstat(fileno(f), &st)) return -1;
    id->volume = (uint64_t)st.st_dev;
    id->index  = (uint64_t)st.st_ino;
    return 0;
#endif
}

int bench_file_id_path(const char *path, bench_file_id_t *id)
{
#ifdef _WIN32
    /* FILE_FLAG_BACKUP_SEMANTICS so a directory can be asked too; the access
     * mask is 0 because we want metadata only and must not disturb the
     * writer holding the file open. Preserve ENOENT: the caller distinguishes
     * "gone" (safe) from "cannot tell" (refuse). */
    HANDLE h = CreateFileA(path, 0,
                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           NULL, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, NULL);
    int rc;
    if (h == INVALID_HANDLE_VALUE) {
        const DWORD e = GetLastError();
        errno = (e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND)
                ? ENOENT : EACCES;
        return -1;
    }
    rc = bench_file_id_from_handle(h, id);
    CloseHandle(h);
    return rc;
#else
    struct stat st;
    if (stat(path, &st)) return -1;
    id->volume = (uint64_t)st.st_dev;
    id->index  = (uint64_t)st.st_ino;
    return 0;
#endif
}

/* An all-zero id is "the filesystem would not say" -- a FAT volume, or a
 * network redirector that does not supply a file index. It is not an identity
 * and must never be compared as one. */
int bench_file_id_valid(const bench_file_id_t *id)
{
    return id->volume != 0 || id->index != 0;
}

int bench_file_id_equal(const bench_file_id_t *a, const bench_file_id_t *b)
{
    return bench_file_id_valid(a) && bench_file_id_valid(b) &&
           a->volume == b->volume && a->index == b->index;
}

int bench_stat_path(const char *path, bench_stat_t *st)
{
#ifdef _WIN32
    return _stat64(path, st);
#else
    return stat(path, st);
#endif
}

int bench_is_regular_mode(unsigned short mode)
{
#ifdef _WIN32
    return (mode & _S_IFMT) == _S_IFREG;
#else
    return S_ISREG(mode);
#endif
}

int bench_stdout_is_tty(void)
{
#ifdef _WIN32
    return _isatty(_fileno(stdout));
#else
    return isatty(fileno(stdout));
#endif
}

int bench_path_exists(const char *path)
{
#ifdef _WIN32
    return _access(path, 0) == 0;
#else
    return access(path, F_OK) == 0;
#endif
}

int bench_atomic_replace(const char *src, const char *dst)
{
#ifdef _WIN32
    if (MoveFileExA(src, dst, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        return 0;
    {
        const DWORD e = GetLastError();
        if (e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND) errno = ENOENT;
        else if (e == ERROR_ACCESS_DENIED || e == ERROR_SHARING_VIOLATION) errno = EACCES;
        else if (e == ERROR_ALREADY_EXISTS || e == ERROR_FILE_EXISTS) errno = EEXIST;
        else errno = EIO;
    }
    return -1;
#else
    return rename(src, dst);
#endif
}

int bench_sync_parent(const char *path)
{
#ifdef _WIN32
    /* bench_atomic_replace() requests write-through from MoveFileEx. Windows
     * does not expose the POSIX "open directory then fsync(fd)" operation. */
    (void)path;
    return 0;
#else
    /* Sized from the longest path the callers can hand over, not a constant
     * that happens to equal it today: every caller's buffer is CKPT_PATH_MAX,
     * and a hardcoded 2048 would start returning ENAMETOOLONG the moment that
     * grew. The parent is always shorter than the path itself. */
    char dir[BENCH_PATH_MAX];
    const char *slash = strrchr(path, '/');
    int fd, rc, saved;
    if (!slash) {
        strcpy(dir, ".");
    } else if (slash == path) {
        strcpy(dir, "/");
    } else {
        const size_t n = (size_t)(slash - path);
        if (n >= sizeof dir) { errno = ENAMETOOLONG; return -1; }
        memcpy(dir, path, n);
        dir[n] = '\0';
    }
    fd = open(dir, O_RDONLY);
    if (fd < 0) return -1;
    rc = fsync(fd);
    saved = errno;
    if (close(fd) && !rc) { rc = -1; saved = errno; }
    errno = saved;
    return rc;
#endif
}

int bench_lock_create(const char *path)
{
#ifdef _WIN32
    return _open(path, _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY,
                 _S_IREAD | _S_IWRITE);
#else
    return open(path, O_WRONLY | O_CREAT | O_EXCL, 0644);
#endif
}

int bench_process_exists(long pid)
{
#ifdef _WIN32
    HANDLE h;
    DWORD code = 0;
    if (pid <= 0) return 0;
    h = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION,
                    FALSE, (DWORD)pid);
    if (!h) return GetLastError() == ERROR_ACCESS_DENIED;
    if (!GetExitCodeProcess(h, &code)) { CloseHandle(h); return 1; }
    CloseHandle(h);
    return code == STILL_ACTIVE;
#else
    if (pid <= 0) return 0;
    if (kill((pid_t)pid, 0) == 0) return 1;
    return errno == EPERM;
#endif
}

long bench_getpid(void)
{
#ifdef _WIN32
    return (long)_getpid();
#else
    return (long)getpid();
#endif
}

int bench_fd_write(int fd, const void *buf, size_t n)
{
    if (n > (size_t)INT_MAX) { errno = EOVERFLOW; return -1; }
#ifdef _WIN32
    return _write(fd, buf, (unsigned)n);
#else
    return (int)write(fd, buf, n);
#endif
}

int bench_fd_close(int fd)
{
#ifdef _WIN32
    return _close(fd);
#else
    return close(fd);
#endif
}

int bench_localtime(const time_t *t, struct tm *out)
{
#ifdef _WIN32
    return localtime_s(out, t) == 0 ? 0 : -1;
#else
    return localtime_r(t, out) ? 0 : -1;
#endif
}

/* UTC. runlog_stamp uses this on Windows, where MSVC's strftime has no %z and
 * a local timestamp could not carry its offset -- an unambiguous UTC stamp
 * beats a local one whose zone is not recorded. */
int bench_gmtime(const time_t *t, struct tm *out)
{
#ifdef _WIN32
    return gmtime_s(out, t) == 0 ? 0 : -1;
#else
    return gmtime_r(t, out) ? 0 : -1;
#endif
}

int64_t bench_getline(char **line, size_t *cap, FILE *f)
{
#ifdef _WIN32
    size_t n = 0;
    int ch;
    if (!line || !cap || !f) { errno = EINVAL; return -1; }
    if (!*line || *cap < 2) {
        size_t initial = *cap >= 2 ? *cap : 256;
        char *p = (char *)realloc(*line, initial);
        if (!p) { errno = ENOMEM; return -1; }
        *line = p;
        *cap = initial;
    }
    while ((ch = fgetc(f)) != EOF) {
        if (n + 1 >= *cap) {
            size_t next = *cap > SIZE_MAX / 2 ? 0 : *cap * 2;
            char *p;
            if (!next) { errno = EOVERFLOW; return -1; }
            p = (char *)realloc(*line, next);
            if (!p) { errno = ENOMEM; return -1; }
            *line = p;
            *cap = next;
        }
        (*line)[n++] = (char)ch;
        if (ch == '\n') break;
    }
    if (!n && ch == EOF) return -1;
    (*line)[n] = '\0';
    return (int64_t)n;
#else
    return (int64_t)getline(line, cap, f);
#endif
}

/* Exit without running atexit handlers or flushing stdio.
 *
 * Called from the clean-stop callback, which is signal context on POSIX, and
 * _exit is the async-signal-safe way out; flushing stdio from a handler is
 * not. MSVC provides _exit with the same semantics, so there is no #ifdef
 * here -- the one this replaces had two byte-identical branches, which told a
 * reader the platforms differed when they do not. */
void bench_fast_exit(int code)
{
    _exit(code);
}

/* ---- clean-stop hook ---------------------------------------------------- */

static bench_stop_handler_t g_stop_fn = NULL;

#ifdef _WIN32
static BOOL WINAPI bench_console_ctrl(DWORD type)
{
    switch (type) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
    case CTRL_LOGOFF_EVENT:
    case CTRL_SHUTDOWN_EVENT:
        if (g_stop_fn) g_stop_fn();
        /* TRUE: handled, so the CRT's own handler further down the chain does
         * not also raise SIGINT and count the same request twice. */
        return TRUE;
    default:
        return FALSE;
    }
}
#else
static void (*g_old_int)(int)  = SIG_ERR;
static void (*g_old_term)(int) = SIG_ERR;

static void bench_on_signal(int sig)
{
    (void)sig;
    if (g_stop_fn) g_stop_fn();
}
#endif

int bench_stop_hook_install(bench_stop_handler_t fn)
{
    g_stop_fn = fn;
#ifdef _WIN32
    return SetConsoleCtrlHandler(bench_console_ctrl, TRUE) ? 0 : -1;
#else
    g_old_int  = signal(SIGINT,  bench_on_signal);
    g_old_term = signal(SIGTERM, bench_on_signal);
    return (g_old_int == SIG_ERR && g_old_term == SIG_ERR) ? -1 : 0;
#endif
}

void bench_stop_hook_remove(void)
{
#ifdef _WIN32
    SetConsoleCtrlHandler(bench_console_ctrl, FALSE);
#else
    if (g_old_int  != SIG_ERR) signal(SIGINT,  g_old_int);
    if (g_old_term != SIG_ERR) signal(SIGTERM, g_old_term);
    g_old_int = g_old_term = SIG_ERR;
#endif
    g_stop_fn = NULL;
}

/* ---- run-time dynamic loading ------------------------------------------ */

void *bench_dlopen(const char *name)
{
#ifdef _WIN32
    return (void *)LoadLibraryA(name);
#else
    return dlopen(name, RTLD_LAZY | RTLD_LOCAL);
#endif
}

void bench_dlsym(void *lib, const char *name, void *slot)
{
#ifdef _WIN32
    FARPROC p = GetProcAddress((HMODULE)lib, name);
#else
    void *p = dlsym(lib, name);
#endif
    memcpy(slot, &p, sizeof p);
}

void bench_dlclose(void *lib)
{
    if (!lib) return;
#ifdef _WIN32
    FreeLibrary((HMODULE)lib);
#else
    dlclose(lib);
#endif
}
