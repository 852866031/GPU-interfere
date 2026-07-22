// GPU colocation probe harness (no NCU / no perf-counter permissions needed).
//
// Characterizes a kernel by MEASURING how much it slows down when co-run with a
// canonical "antagonist" that saturates one shared resource. Also reports static
// SM-residency info via the CUDA occupancy API (no counters required).
//
// Registry of workloads (each = a kernel that stresses ~one resource):
//   sleep : pure scheduler / occupancy, no memory or compute
//   dram  : streaming copy of a large (>L2) array   -> DRAM bandwidth
//   l2    : streaming copy of a 16 MB array (L2-resident) -> L2 bandwidth
//   l1    : per-block copy of an L1-sized region    -> L1 cache
//   fma   : FP32 fused-multiply-add, high ILP        -> FMA pipeline + warp scheduler
//   fp64  : FP64 fused-multiply-add, high ILP        -> FP64 pipeline
//
// Usage:
//   probe list
//   probe static <name>                 -> STATIC name=.. regs=.. smem=.. maxblocks=.. ...
//   probe alone  <name>                 -> ALONE  name=.. ms=..
//   probe coloc  <target> <antagonist>  -> COLOC target=.. antag=.. alone_ms=.. coloc_ms=.. slowdown=..
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <cuda_runtime.h>
using namespace std;

#define CK(cmd) do{ cudaError_t e=cmd; if(e!=cudaSuccess){ printf("CUDA error %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

// ---- block->SM trace: which SM each block ran on, entry/exit in globaltimer ns ----
struct BlockEv { unsigned smid, blk; unsigned long long t0, t1; };
#define TRACE_BEGIN unsigned long long _t0=0; unsigned _smid=0; \
  if(trace && threadIdx.x==0){ \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_t0)); \
    asm volatile("mov.u32 %0, %%smid;"        : "=r"(_smid)); }
#define TRACE_END if(trace && threadIdx.x==0){ \
    unsigned long long _t1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_t1)); \
    trace[blockIdx.x].smid=_smid; trace[blockIdx.x].blk=blockIdx.x; \
    trace[blockIdx.x].t0=_t0; trace[blockIdx.x].t1=_t1; }

// ------------------------- kernels -------------------------
__global__ void k_sleep(long long it, BlockEv* trace){
    TRACE_BEGIN
    for(long long i=0;i<it;i++) asm volatile("nanosleep.u32 1000;");
    TRACE_END
}
__global__ void k_copy(const float* in, float* out, long long n, long long it, BlockEv* trace){
    TRACE_BEGIN
    size_t s = blockIdx.x*blockDim.x + threadIdx.x, st = (size_t)gridDim.x*blockDim.x;
    for(long long j=0;j<it;j++) for(size_t i=s;i<(size_t)n;i+=st) out[i]=in[i];
    TRACE_END
}
__global__ void k_copy_tb(const float* in, float* out, long long npb, long long it, int region_bytes, BlockEv* trace){
    TRACE_BEGIN
    int fpr = region_bytes/4;
    int rpt = (npb + fpr - 1)/fpr;
    int b0 = blockIdx.x*rpt*fpr, be = b0 + npb;
    for(long long i=0;i<it;i++) for(int j=b0+threadIdx.x;j<be;j+=blockDim.x) out[j]=in[j];
    TRACE_END
}
__global__ void k_fma32(const float* a, const float* b, float* c, long long it, BlockEv* trace){
    TRACE_BEGIN
    float o1=a[threadIdx.x],o2=b[threadIdx.x],x=0,y=0,z=0,w=0;
    for(long long i=0;i<it;i++){ x=__fmaf_rn(o1,o2,x); y=__fmaf_rn(o1,o2,y); z=__fmaf_rn(o1,o2,z); w=__fmaf_rn(o1,o2,w);}
    c[threadIdx.x]=x+y+z+w;
    TRACE_END
}
__global__ void k_fma64(const double* a, const double* b, double* c, long long it, BlockEv* trace){
    TRACE_BEGIN
    double o1=a[threadIdx.x],o2=b[threadIdx.x],x=0,y=0,z=0,w=0;
    for(long long i=0;i<it;i++){ x=__fma_rn(o1,o2,x); y=__fma_rn(o1,o2,y); z=__fma_rn(o1,o2,z); w=__fma_rn(o1,o2,w);}
    c[threadIdx.x]=x+y+z+w;
    TRACE_END
}

// ------------------------- workload wrapper -------------------------
struct Workload {
    string name;
    int blocks, threads;
    function<void(cudaStream_t, BlockEv*)> launch;   // one timed iteration (trace=nullptr normally)
    const void* func;                                // for occupancy API
    int smem = 0;
};

static int g_sms = 0;

// build a workload by name; allocates its device buffers (leaked intentionally; short-lived tool)
Workload make(const string& name){
    Workload w; w.name=name; w.blocks=g_sms;
    if(name=="sleep"){
        long long it=100000; w.threads=768; w.func=(const void*)k_sleep;
        w.launch=[=](cudaStream_t s, BlockEv* tr){ k_sleep<<<g_sms,768,0,s>>>(it,tr); };
    } else if(name=="dram"){
        long long n=128LL*1024*1024; long long it=48; w.threads=512; w.func=(const void*)k_copy; // 512 MB
        float *in,*out; CK(cudaMalloc(&in,n*4)); CK(cudaMalloc(&out,n*4));
        w.launch=[=](cudaStream_t s, BlockEv* tr){ k_copy<<<g_sms,512,0,s>>>(in,out,n,it,tr); };
    } else if(name=="l2"){
        long long n=4LL*1024*1024; long long it=1200; w.threads=512; w.func=(const void*)k_copy;   // 16 MB (L2-resident)
        float *in,*out; CK(cudaMalloc(&in,n*4)); CK(cudaMalloc(&out,n*4));
        w.launch=[=](cudaStream_t s, BlockEv* tr){ k_copy<<<g_sms,512,0,s>>>(in,out,n,it,tr); };
    } else if(name=="l1"){
        long long npb=8192; long long it=15000; int rb=128*1024; w.threads=64; w.func=(const void*)k_copy_tb; // 32 KB/block
        long long tot=(long long)g_sms*(rb/4)*((npb+rb/4-1)/(rb/4));
        float *in,*out; CK(cudaMalloc(&in,tot*4)); CK(cudaMalloc(&out,tot*4));
        w.launch=[=](cudaStream_t s, BlockEv* tr){ k_copy_tb<<<g_sms,64,0,s>>>(in,out,npb,it,rb,tr); };
    } else if(name=="fma"){
        long long it=20000000; w.threads=128; w.func=(const void*)k_fma32;
        float *a,*b,*c; CK(cudaMalloc(&a,128*4)); CK(cudaMalloc(&b,128*4)); CK(cudaMalloc(&c,128*4));
        w.launch=[=](cudaStream_t s, BlockEv* tr){ k_fma32<<<g_sms,128,0,s>>>(a,b,c,it,tr); };
    } else if(name=="fp64"){
        long long it=400000; w.threads=128; w.func=(const void*)k_fma64;
        double *a,*b,*c; CK(cudaMalloc(&a,128*8)); CK(cudaMalloc(&b,128*8)); CK(cudaMalloc(&c,128*8));
        w.launch=[=](cudaStream_t s, BlockEv* tr){ k_fma64<<<g_sms,128,0,s>>>(a,b,c,it,tr); };
    } else { printf("unknown workload %s\n",name.c_str()); exit(1); }
    return w;
}

double median(vector<double>& v){ sort(v.begin(),v.end()); return v[v.size()/2]; }

double timeAlone(Workload& w, int reps){
    cudaStream_t s; CK(cudaStreamCreate(&s));
    w.launch(s,nullptr); CK(cudaStreamSynchronize(s)); // warm up
    vector<double> lat;
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for(int i=0;i<reps;i++){ CK(cudaEventRecord(a,s)); w.launch(s,nullptr); CK(cudaEventRecord(b,s)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms,a,b)); lat.push_back(ms); }
    CK(cudaStreamDestroy(s));
    return median(lat);
}

// time target on stream A while a batch of antagonist launches keeps stream B busy throughout
double timeColoc(Workload& tgt, Workload& ant, int reps, double tgtAlone, double antAlone){
    cudaStream_t sa,sb; CK(cudaStreamCreateWithFlags(&sa,cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&sb,cudaStreamNonBlocking));
    int nBatch = (int)((reps*tgtAlone)/antAlone) + 4;   // ensure antagonist outlasts the target measurement
    for(int i=0;i<nBatch;i++) ant.launch(sb,nullptr);            // queue antagonist work (do NOT wait)
    vector<double> lat;
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for(int i=0;i<reps;i++){ CK(cudaEventRecord(a,sa)); tgt.launch(sa,nullptr); CK(cudaEventRecord(b,sa)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms,a,b)); lat.push_back(ms); }
    CK(cudaDeviceSynchronize());
    CK(cudaStreamDestroy(sa)); CK(cudaStreamDestroy(sb));
    return median(lat);
}

int main(int argc,char** argv){
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0)); g_sms=p.multiProcessorCount;
    if(argc<2){ printf("usage: probe list|static|alone|coloc ...\n"); return 1; }
    string cmd=argv[1];
    const char* names[]={"sleep","dram","l2","l1","fma","fp64"};

    if(cmd=="list"){ for(auto n:names) printf("%s\n",n); return 0; }

    if(cmd=="static"){
        Workload w=make(argv[2]);
        cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa,w.func));
        int maxblocks=0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxblocks,w.func,w.threads,w.smem));
        int warps_per_block=(w.threads+31)/32;
        double occ = (maxblocks*warps_per_block) / (double)(p.maxThreadsPerMultiProcessor/32);
        printf("STATIC name=%s regs=%d smem=%zu threads=%d warps_per_block=%d max_blocks_per_sm=%d occupancy=%.3f sms=%d\n",
               w.name.c_str(), fa.numRegs, fa.sharedSizeBytes, w.threads, warps_per_block, maxblocks, occ, g_sms);
        return 0;
    }
    if(cmd=="alone"){
        Workload w=make(argv[2]); double ms=timeAlone(w,7);
        printf("ALONE name=%s ms=%.4f\n", w.name.c_str(), ms); return 0;
    }
    if(cmd=="once"){   // exactly one launch, for clean NCU profiling (--launch-count 1)
        Workload w=make(argv[2]); cudaStream_t s; CK(cudaStreamCreate(&s));
        w.launch(s,nullptr); CK(cudaStreamSynchronize(s)); return 0;
    }
    if(cmd=="coloc"){
        Workload t=make(argv[2]), a=make(argv[3]);
        double ta=timeAlone(t,7), aa=timeAlone(a,7);
        double tc=timeColoc(t,a,7,ta,aa);
        printf("COLOC target=%s antag=%s alone_ms=%.4f coloc_ms=%.4f slowdown=%.3f\n",
               t.name.c_str(), a.name.c_str(), ta, tc, tc/ta);
        return 0;
    }
    if(cmd=="trace"){   // trace <target> [antagonist] : one traced launch each, colocated
        Workload t=make(argv[2]);
        bool hasA = argc>3;
        Workload a; BlockEv *bt,*ba=nullptr;
        CK(cudaMalloc(&bt,t.blocks*sizeof(BlockEv))); CK(cudaMemset(bt,0,t.blocks*sizeof(BlockEv)));
        if(hasA){ a=make(argv[3]); CK(cudaMalloc(&ba,a.blocks*sizeof(BlockEv))); CK(cudaMemset(ba,0,a.blocks*sizeof(BlockEv))); }
        cudaStream_t sa,sb; CK(cudaStreamCreateWithFlags(&sa,cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&sb,cudaStreamNonBlocking));
        t.launch(sa,nullptr); if(hasA) a.launch(sb,nullptr); CK(cudaDeviceSynchronize());   // warm up, untraced
        if(hasA) a.launch(sb,ba);                            // traced colocated launches (one each)
        t.launch(sa,bt);
        CK(cudaDeviceSynchronize());
        auto dump=[&](Workload& w, BlockEv* d){
            vector<BlockEv> h(w.blocks);
            CK(cudaMemcpy(h.data(),d,w.blocks*sizeof(BlockEv),cudaMemcpyDeviceToHost));
            for(auto& e:h) printf("TRACEBLK kernel=%s block=%u smid=%u t0=%llu t1=%llu\n",
                                  w.name.c_str(), e.blk, e.smid, e.t0, e.t1);
        };
        dump(t,bt); if(hasA) dump(a,ba);
        return 0;
    }
    printf("unknown command %s\n",cmd.c_str()); return 1;
}
