#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>

unsigned int my_rand()
{
    // our initial starting seed is 5323
    static unsigned int nSeed = 5323;

    // Take the current seed and generate a new value from it
    // Due to our use of large constants and overflow, it would be
    // very hard for someone to predict what the next number is
    // going to be from the previous one.
    nSeed = (8253729 * nSeed + 2396403); 

    // Take the seed and return a value between 0 and 100
    return nSeed % 101;
}

SEC("xdp_drop")
int xdp_drop_prog(struct xdp_md *ctx)
{
    __u32 data_end = ctx->data_end;
    __u32 data_start = ctx->data;
    __u32 len = data_end - data_start;

    __u32 prob = (40 * len) / 50000;   

    if(my_rand() <= prob) {
        return XDP_DROP;
    } else {
        return XDP_PASS;
    }
}

char _license[] SEC("license") = "GPL";