"""FP64 kernels for bounded, device-resident LiDAR odometry.

Compile with --fmad=false to retain the native arithmetic order.
Voxel centroids sum in input order; uncertified grid queries scan every target.
"""

_VOXEL_SOURCE = r"""
extern "C" __global__ void centroids(const double* xyz,const int* heads,const int* next,
    const long long* occupied,const long long* order,int groups,double q,double* result,
        int* invalid) {
    int group=blockDim.x*blockIdx.x+threadIdx.x;if(group>=groups)return;
    int slot=(int)occupied[order[group]];int rows[32];int count=0;
    #pragma unroll 1
    for(int row=heads[slot];row>=0;row=next[row]) {
        if(count==32){atomicExch(invalid,1);return;}
        int j=count;while(j>0&&rows[j-1]>row){rows[j]=rows[j-1];--j;}
        rows[j]=row;++count;
    }
    #pragma unroll 1
    for(int a=0;a<3;++a) {
        double sum=0.;for(int i=0;i<count;++i)sum+=xyz[3*rows[i]+a];
        double v=sum/(double)count;if(q>0)v=nearbyint(v/q)*q;
        result[3*group+a]=v;
    }
}

extern "C" __global__ void prepare_crop(const double* input,int n,double q,double voxel,
    double low,double high,double* xyz,unsigned long long* keys,int* state) {
    int row=blockDim.x*blockIdx.x+threadIdx.x;if(row>=n)return;
    long long k[3];double p[3];bool invalid=false;
    for(int a=0;a<3;++a) {
        double v=input[3*row+a];if(q>0)v=nearbyint(v/q)*q;
        p[a]=v;xyz[3*row+a]=v;double f=floor(v/voxel);
        if(!isfinite(v)||f< -1048576.||f>=1048576.){invalid=true;f=0.;}
        k[a]=(long long)f+1048576;
    }
    if(invalid)atomicExch(state,1);
    double range=sqrt((p[0]*p[0]+p[1]*p[1])+p[2]*p[2]);
    keys[row]=(range>=low && range<=high)?
        ((((unsigned long long)k[0]<<21)|(unsigned long long)k[1])<<21
         |(unsigned long long)k[2]):~0ULL;
}
extern "C" __global__ void insert_masked(const unsigned long long* keys,int n,int mask,
    unsigned long long* table,int* heads,int* next) {
    int row=blockDim.x*blockIdx.x+threadIdx.x;if(row>=n)return;
    unsigned long long key=keys[row];if(key==~0ULL)return;
    unsigned long long h=key;h^=h>>30;h*=0xbf58476d1ce4e5b9ULL;h^=h>>27;h*=0x94d049bb133111ebULL;
        h^=h>>31;
    int slot=(int)h&mask;
    while(true){unsigned long long old=atomicCAS(table+slot,~0ULL,key);
        if(old==~0ULL||old==key)break;slot=(slot+1)&mask;}
    next[row]=atomicExch(heads+slot,row);
}
extern "C" __global__ void compact_cells(const unsigned long long* table,const int* heads,int n,
    long long* occupied,unsigned long long* compact_keys,int* state) {
    int row=blockDim.x*blockIdx.x+threadIdx.x;
    unsigned int active=__ballot_sync(0xffffffff,row<n && heads[row]>=0);
    if(!active)return;
    int lane=threadIdx.x&31;
    int begin=lane==0?atomicAdd(state+1,__popc(active)):0;
    begin=__shfl_sync(0xffffffff,begin,0);
    if(row<n && heads[row]>=0){int pos=begin+__popc(active&((1u<<lane)-1));occupied[pos]=row;
        compact_keys[pos]=table[row];}
}
"""

_GYRO_SOURCE = r"""
__device__ void matrix_vector(const double* a,const double* x,double* y,bool transpose=false){
    for(int i=0;i<3;i++) y[i]=(transpose?a[i]:a[3*i])*x[0]
        +(transpose?a[3+i]:a[3*i+1])*x[1]+(transpose?a[6+i]:a[3*i+2])*x[2];
}
__device__ void rotvec_apply(const double* v,const double* p,double* out){
    double angle=sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    double square=angle*angle;
    double scale=angle<=1e-3?0.5-square/48.0+(square*square)/3840.0:sin(0.5*angle)/angle;
    double x=v[0]*scale,y=v[1]*scale,z=v[2]*scale,w=cos(0.5*angle);
    double matrix[9]={x*x-y*y-z*z+w*w,2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),-x*x+y*y-z*z+w*w,2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),-x*x-y*y+z*z+w*w};
    matrix_vector(matrix,p,out);
}
extern "C" __global__ void gyro_rotate3(
    const double* points,const double* query,const double* times,
    const double* orientations,const double* rates,const double* projected,
    const double* mounting,const double* back,const double* start_sum,
    const int n,const int m,double* output){
    int row=blockDim.x*blockIdx.x+threadIdx.x;if(row>=n)return;
    int low=0,high=m;
    while(low<high){int middle=low+(high-low)/2;if(times[middle]<=query[row])low=middle+1;
        else high=middle;}
    int index=max(0,min(m-2,low-1));
    double elapsed=query[row]-times[index],fraction=elapsed/(times[index+1]-times[index]);
    double p[3],partial[3],world[3],corrected[3],v[3];
    matrix_vector(mounting,points+3*row,p,true);
    for(int a=0;a<3;a++)v[a]=rates[3*index+a]*elapsed;
    rotvec_apply(v,p,partial);matrix_vector(orientations+9*index,partial,world);
    for(int a=0;a<3;
        a++)v[a]=-(projected[3*index+
        a]+
        fraction*(projected[3*(index+
        1)+
        a]-projected[3*index+
        a])-start_sum[a]);
    rotvec_apply(v,world,corrected);matrix_vector(back,corrected,output+3*row);
}
"""

_GEOMETRY_SOURCE = r"""
extern "C" __global__ void transform_points(const double* xyz,const double* matrix,int n,
        double quantum,double* output) {
    int row=blockDim.x*blockIdx.x+threadIdx.x;if(row>=n)return;
    for(int a=0;a<3;
        ++a){double v=(matrix[4*a]*xyz[3*row]+
        matrix[4*a+
        1]*xyz[3*row+
        1])+
        matrix[4*a+
        2]*xyz[3*row+
        2]+
        matrix[4*a+
        3];
        if(quantum>0)v=nearbyint(v/quantum)*quantum;output[3*row+a]=v;}
}
extern "C" __global__ void translate_points(const double* xyz,const double* offsets,
        const double* translation,int n,double gap,double* output) {
    int row=blockDim.x*blockIdx.x+threadIdx.x;if(row>=n)return;
    double fraction=offsets[row]/gap;
    for(int a=0;a<3;++a)output[3*row+a]=xyz[3*row+a]+fraction*translation[a];
}
extern "C" __global__ void rigid_deskew(const double* xyz,const double* offsets,
        const double* vector,const double* translation,int n,double gap,double* output) {
    int row=blockDim.x*blockIdx.x+threadIdx.x;if(row>=n)return;
    double fraction=offsets[row]/gap,v[3],p[3];
    for(int a=0;a<3;++a)v[a]=vector[a]*fraction;
    rotvec_apply(v,xyz+3*row,p);
    for(int a=0;a<3;++a)output[3*row+a]=p[a]+fraction*translation[a];
}
"""

_GRID_SOURCE = r"""
__device__ unsigned long long grid_mix(unsigned long long h) {
    h^=h>>30;h*=0xbf58476d1ce4e5b9ULL;h^=h>>27;h*=0x94d049bb133111ebULL;return h^(h>>31);
}
__device__ unsigned long long grid_key(long long x,long long y,long long z) {
    return (((unsigned long long)(x+
        1048576)<<21)|(unsigned long long)(y+
        1048576))<<21|(unsigned long long)(z+
        1048576);
}
__device__ __forceinline__ void insert_best(double d,long long id,int k,double* best,
        long long* chosen) {
    if(d<best[k-1] || (d==best[k-1] && id<chosen[k-1])) {
        int at=k-1;
        while(at>0 && (d<best[at-1] || (d==best[at-1] && id<chosen[at-1]))) {
            best[at]=best[at-1];chosen[at]=chosen[at-1];--at;
        }
        best[at]=d;chosen[at]=id;
    }
}
__device__ __forceinline__ void nearest_grid(
    const double* p,const double* reference,const long long* cell_keys,
    const int* cell_heads,const int* next_rows,const int* mask,
    long long* brute,const double* info,int m,int k,int row,double* best,long long* chosen) {
    brute[row+1]=0;
    double width=info[0];
    double error=128*2.2204460492503131e-16*fmax(info[1],fmax(1.,fmax(fabs(p[0]),fmax(fabs(p[1]),
        fabs(p[2])))));
    long long cell[3];bool bounded=true;
    for(int a=0;a<3;++a) {
        double v=floor(p[a]/width);bounded&=isfinite(v)&&fabs(v)<1048572.;
        cell[a]=bounded?(long long)v:0;
    }
    for(int j=0;j<k;++j){best[j]=1.0/0.0;chosen[j]=m;}
    if(bounded) for(int ring=0;ring<=3;++ring) {
        for(int x=-ring;x<=ring;++x)for(int y=-ring;y<=ring;++y)for(int z=-ring;z<=ring;++z) {
            if(ring && abs(x)!=ring && abs(y)!=ring && abs(z)!=ring)continue;
            long long c[3]={cell[0]+x,cell[1]+y,cell[2]+z};double lower=0;
            for(int a=0;a<3;++a){double d=fmax(0.,fmax(c[a]*width-p[a],
        p[a]-(c[a]+1)*width)-error);lower+=d*d;}
            if(lower>best[k-1])continue;
            unsigned long long key=grid_key(c[0],c[1],c[2]);int slot=grid_mix(key)&mask[0];
            while((unsigned long long)cell_keys[slot]!=key && cell_keys[slot]!=-1)slot=(slot+
        1)&mask[0];
            if(cell_keys[slot]==-1)continue;
            for(int id=cell_heads[slot];id>=0;id=next_rows[id]) {
                double dx=p[0]-reference[3*id],dy=p[1]-reference[3*id+1],dz=p[2]-reference[3*id+2];
                insert_best((dx*dx+dy*dy)+dz*dz,id,k,best,chosen);
            }
        }
        double edge=1.0/0.0;
        for(int a=0;a<3;++a)edge=fmin(edge,fmin(p[a]-(cell[a]-ring)*width,
        (cell[a]+ring+1)*width-p[a]));
        if(isfinite(best[k-1]) && sqrt(best[k-1])+error<fmax(0.,edge-error))return;
    }
    atomicAdd((unsigned long long*)brute,1ULL);
    brute[row+1]=1;
}

extern "C" __global__ void build_grid(const double* xyz,int n,double width,
    long long* keys,int* heads,int* next,int mask,int* invalid) {
    int row=blockDim.x*blockIdx.x+threadIdx.x;if(row>=n)return;
    long long c[3];
    for(int a=0;a<3;++a){double v=floor(xyz[3*row+a]/width);
        if(!isfinite(v)||fabs(v)>=1048572.){atomicExch(invalid,1);return;}c[a]=(long long)v;}
    unsigned long long key=grid_key(c[0],c[1],c[2]);int p=grid_mix(key)&mask;
    while(true){unsigned long long old=atomicCAS((unsigned long long*)(keys+p),~0ULL,key);
        if(old==~0ULL||old==key)break;p=(p+1)&mask;}
    next[row]=atomicExch(heads+p,row);
}
extern "C" __global__ void query_grid(const double* xyz,const double* reference,
    const long long* keys,const int* heads,const int* next,const int* mask,
    long long* brute,const double* info,int n,int m,int k,double* ds,long long* ids) {
    int row=blockDim.x*blockIdx.x+threadIdx.x;if(row>=n)return;
    double best[17];long long chosen[17];
    nearest_grid(xyz+3*row,reference,keys,heads,next,mask,brute,info,m,k,row,best,chosen);
    for(int j=0;j<k;++j){ds[row*k+j]=sqrt(best[j]);ids[row*k+j]=chosen[j];}
}
extern "C" __global__ void parallel_query(const double* xyz,const double* reference,
    const long long* keys,const int* heads,const int* next,const int* mask,
    long long* brute,const double* info,int n,int m,int requested_k,double* ds,long long* ids) {

    int row=blockIdx.x;
    if(row>=n || brute[row+1]==0)return;
    const int tid=threadIdx.x;
    const int k=requested_k;
    double p[3]={xyz[3*row],xyz[3*row+1],xyz[3*row+2]};
    double local_best[17];long long local_chosen[17];
    for(int j=0;j<k;++j){local_best[j]=1.0/0.0;local_chosen[j]=m;}
    for(int id=tid;id<m;id+=128) {
        double dx=p[0]-reference[3*id],dy=p[1]-reference[3*id+1],dz=p[2]-reference[3*id+2];
        insert_best((dx*dx+dy*dy)+dz*dz,id,k,local_best,local_chosen);
    }
    __shared__ double shared_best[128*17];
    __shared__ long long shared_chosen[128*17];
    for(int j=0;j<k;++j){shared_best[17*tid+j]=local_best[j];
        shared_chosen[17*tid+j]=local_chosen[j];}
    __syncthreads();
    for(int stride=64;stride>0;stride/=2) {
        if(tid<stride)for(int j=0;j<k;++j)
            insert_best(shared_best[17*(tid+stride)+j],shared_chosen[17*(tid+stride)+j],k,
                        shared_best+17*tid,shared_chosen+17*tid);
        __syncthreads();
    }
    if(tid!=0)return;

    for(int j=0;j<k;++j){ds[row*k+j]=sqrt(shared_best[j]);ids[row*k+j]=shared_chosen[j];}
}
"""

_FUSED_GRID_SOURCE = r"""
__device__ unsigned long long grid_mix(unsigned long long h) {
    h^=h>>30;h*=0xbf58476d1ce4e5b9ULL;h^=h>>27;h*=0x94d049bb133111ebULL;return h^(h>>31);
}
__device__ unsigned long long grid_key(long long x,long long y,long long z) {
    return (((unsigned long long)(x+
        1048576)<<21)|(unsigned long long)(y+
        1048576))<<21|(unsigned long long)(z+
        1048576);
}
__device__ __forceinline__ void insert_best(double d,long long id,int k,double* best,
        long long* chosen) {
    if(d<best[k-1] || (d==best[k-1] && id<chosen[k-1])) {
        int at=k-1;
        while(at>0 && (d<best[at-1] || (d==best[at-1] && id<chosen[at-1]))) {
            best[at]=best[at-1];chosen[at]=chosen[at-1];--at;
        }
        best[at]=d;chosen[at]=id;
    }
}
__device__ __forceinline__ void nearest_grid(
    const double* p,const double* reference,const long long* cell_keys,
    const int* cell_heads,const int* next_rows,const int* mask,
    long long* brute,const double* info,int m,int k,int row,double* best,long long* chosen) {
    brute[row+1]=0;
    double width=info[0];
    double error=128*2.2204460492503131e-16*fmax(info[1],fmax(1.,fmax(fabs(p[0]),fmax(fabs(p[1]),
        fabs(p[2])))));
    long long cell[3];bool bounded=true;
    for(int a=0;a<3;++a) {
        double v=floor(p[a]/width);bounded&=isfinite(v)&&fabs(v)<1048572.;
        cell[a]=bounded?(long long)v:0;
    }
    for(int j=0;j<k;++j){best[j]=1.0/0.0;chosen[j]=m;}
    if(bounded) for(int ring=0;ring<=3;++ring) {
        for(int x=-ring;x<=ring;++x)for(int y=-ring;y<=ring;++y)for(int z=-ring;z<=ring;++z) {
            if(ring && abs(x)!=ring && abs(y)!=ring && abs(z)!=ring)continue;
            long long c[3]={cell[0]+x,cell[1]+y,cell[2]+z};double lower=0;
            for(int a=0;a<3;++a){double d=fmax(0.,fmax(c[a]*width-p[a],
        p[a]-(c[a]+1)*width)-error);lower+=d*d;}
            if(lower>best[k-1])continue;
            unsigned long long key=grid_key(c[0],c[1],c[2]);int slot=grid_mix(key)&mask[0];
            while((unsigned long long)cell_keys[slot]!=key && cell_keys[slot]!=-1)slot=(slot+
        1)&mask[0];
            if(cell_keys[slot]==-1)continue;
            for(int id=cell_heads[slot];id>=0;id=next_rows[id]) {
                double dx=p[0]-reference[3*id],dy=p[1]-reference[3*id+1],dz=p[2]-reference[3*id+2];
                insert_best((dx*dx+dy*dy)+dz*dz,id,k,best,chosen);
            }
        }
        double edge=1.0/0.0;
        for(int a=0;a<3;++a)edge=fmin(edge,fmin(p[a]-(cell[a]-ring)*width,
        (cell[a]+ring+1)*width-p[a]));
        if(isfinite(best[k-1]) && sqrt(best[k-1])+error<fmax(0.,edge-error))return;
    }
    atomicAdd((unsigned long long*)brute,1ULL);
    brute[row+1]=1;
}

extern "C" __global__ void project_query_system3(
    const double* xyz, const double* matrix, const double* reference,
    const long long* cell_keys, const int* cell_heads, const int* next_rows,
    const int* mask, long long* brute, const double* info, const double* normal,
    const bool* planar, const bool* done, const int n, const int m, const double gate,
    const double huber, const double error, double* uncertain, double* terms,
    double* moved_out, double* distances_out, long long* indices_out,
    long long* pending_out) {
    int row=blockDim.x*blockIdx.x+threadIdx.x;
    if(row>=n) return;
    const double* s=xyz+3*row;
    double p[3];
    for(int a=0;a<3;a++)
        p[a]=matrix[4*a]*s[0]+matrix[4*a+1]*s[1]+matrix[4*a+2]*s[2]+matrix[4*a+3];
    double best[2];long long chosen[2];
    nearest_grid(p,reference,cell_keys,cell_heads,next_rows,mask,brute,info,m,2,row,best,chosen);
    double distance=sqrt(best[0]);long long hit=chosen[0];
    for(int a=0;a<3;a++) moved_out[3*row+a]=p[a];
    distances_out[2*row]=distance;distances_out[2*row+1]=sqrt(best[1]);
    indices_out[2*row]=hit;indices_out[2*row+1]=chosen[1];
    bool repair=sqrt(best[1])-distance<=error || fabs(distance-gate)<=error;
    uncertain[row]=repair?1.0:0.0;
    bool valid=distance<gate && hit<m;
    bool pending=valid && !done[hit];
    pending_out[row]=pending?hit:-1;
    if(valid) valid=done[hit] && planar[hit];
    double j[6]={0,0,0,0,0,0},r=0,w=1;
    if(valid){
        const double* q=reference+3*hit;const double* a=normal+3*hit;
        r=(p[0]-q[0])*a[0]+(p[1]-q[1])*a[1]+(p[2]-q[2])*a[2];
        j[0]=p[1]*a[2]-p[2]*a[1];j[1]=p[2]*a[0]-p[0]*a[2];
        j[2]=p[0]*a[1]-p[1]*a[0];j[3]=a[0];j[4]=a[1];j[5]=a[2];
        w=fabs(r)<=huber?1.0:huber/fmax(fabs(r),1e-12);
    }
    int column=0;
    for(int a=0;a<6;a++) for(int b=a;b<6;b++)
        terms[(column++)*n+row]=(j[a]*w)*j[b];
    for(int a=0;a<6;a++) terms[(column++)*n+row]=(j[a]*w)*r;
    terms[27*n+row]=r*r;terms[28*n+row]=valid?1.0:0.0;
    terms[29*n+row]=repair?1.0:0.0;
    terms[30*n+row]=pending?1.0:0.0;
}

extern "C" __global__ void parallel_fused3(
    const double* xyz, const double* matrix, const double* reference,
    const long long* cell_keys, const int* cell_heads, const int* next_rows,
    const int* mask, long long* brute, const double* info, const double* normal,
    const bool* planar, const bool* done, const int n, const int m, const double gate,
    const double huber, const double error, double* uncertain, double* terms,
    double* moved_out, double* distances_out, long long* indices_out,
    long long* pending_out) {

    int row=blockIdx.x;
    if(row>=n || brute[row+1]==0)return;
    const int tid=threadIdx.x;
    const int k=2;
    double p[3]={moved_out[3*row],moved_out[3*row+1],moved_out[3*row+2]};
    double local_best[2];long long local_chosen[2];
    for(int j=0;j<k;++j){local_best[j]=1.0/0.0;local_chosen[j]=m;}
    for(int id=tid;id<m;id+=128) {
        double dx=p[0]-reference[3*id],dy=p[1]-reference[3*id+1],dz=p[2]-reference[3*id+2];
        insert_best((dx*dx+dy*dy)+dz*dz,id,k,local_best,local_chosen);
    }
    __shared__ double shared_best[128*2];
    __shared__ long long shared_chosen[128*2];
    for(int j=0;j<k;++j){shared_best[2*tid+j]=local_best[j];shared_chosen[2*tid+j]=local_chosen[j];}
    __syncthreads();
    for(int stride=64;stride>0;stride/=2) {
        if(tid<stride)for(int j=0;j<k;++j)
            insert_best(shared_best[2*(tid+stride)+j],shared_chosen[2*(tid+stride)+j],k,
                        shared_best+2*tid,shared_chosen+2*tid);
        __syncthreads();
    }
    if(tid!=0)return;
    double best[2]={shared_best[0],shared_best[1]};long long chosen[2]={shared_chosen[0],
        shared_chosen[1]};
    double distance=sqrt(best[0]);long long hit=chosen[0];
    for(int a=0;a<3;a++) moved_out[3*row+a]=p[a];
    distances_out[2*row]=distance;distances_out[2*row+1]=sqrt(best[1]);
    indices_out[2*row]=hit;indices_out[2*row+1]=chosen[1];
    bool repair=sqrt(best[1])-distance<=error || fabs(distance-gate)<=error;
    uncertain[row]=repair?1.0:0.0;
    bool valid=distance<gate && hit<m;
    bool pending=valid && !done[hit];
    pending_out[row]=pending?hit:-1;
    if(valid) valid=done[hit] && planar[hit];
    double j[6]={0,0,0,0,0,0},r=0,w=1;
    if(valid){
        const double* q=reference+3*hit;const double* a=normal+3*hit;
        r=(p[0]-q[0])*a[0]+(p[1]-q[1])*a[1]+(p[2]-q[2])*a[2];
        j[0]=p[1]*a[2]-p[2]*a[1];j[1]=p[2]*a[0]-p[0]*a[2];
        j[2]=p[0]*a[1]-p[1]*a[0];j[3]=a[0];j[4]=a[1];j[5]=a[2];
        w=fabs(r)<=huber?1.0:huber/fmax(fabs(r),1e-12);
    }
    int column=0;
    for(int a=0;a<6;a++) for(int b=a;b<6;b++)
        terms[(column++)*n+row]=(j[a]*w)*j[b];
    for(int a=0;a<6;a++) terms[(column++)*n+row]=(j[a]*w)*r;
    terms[27*n+row]=r*r;terms[28*n+row]=valid?1.0:0.0;
    terms[29*n+row]=repair?1.0:0.0;
    terms[30*n+row]=pending?1.0:0.0;
}
"""
