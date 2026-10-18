"""Strict FP64 native RGB-D kernels; host code normalizes both quaternions."""

SOURCE = r"""

__device__ double weighted(const double* weights,const double* offsets,int row,int probe,
                          int controls,int width,int start,int stride,int compensated) {
    double total=0.0 + weights[row*controls]*offsets[probe*width+start];
    double correction=0.0;
    for(int j=1;j<controls;++j) {
        double term=weights[row*controls+j]*offsets[probe*width+start+stride*j];
        double updated=total+term;
        if(compensated) {
            if(fabs(total)>=fabs(term)) correction+=(total-updated)+term;
            else correction+=(term-updated)+total;
        }
        total=updated;
    }
    if(compensated && correction!=0.0 && isfinite(correction)) total+=correction;
    return total;
}
__device__ void transformed(double x,double y,double z,const double* f,double* out) {
    double qx=f[3],qy=f[4],qz=f[5],qw=f[6],cx=f[7],cy=f[8],cz=f[9],cw=f[10];
    double ax=qw*x+qx*0.0+qy*z-qz*y;
    double ay=qw*y-qx*z+qy*0.0+qz*x;
    double az=qw*z+qx*y-qy*x+qz*0.0;
    double aw=qw*0.0-qx*x-qy*y-qz*z;
    out[0]=aw*cx+ax*cw+ay*cz-az*cy+f[0];
    out[1]=aw*cy-ax*cz+ay*cw+az*cx+f[1];
    out[2]=aw*cz+ax*cy-ay*cx+az*cw+f[2];
}
extern "C" __global__ void fields(int kind,int n,int d,int controls,int width,
 const double* source,const double* depths,const double* weights,const double* target,
 const double* target_weights,const double* planes,const double* normals,const double* sqrt_weights,
 const double* parameters,const double* first,const double* second,const double* steps,
 int compensated,double* output,int* invalid) {
    int index=blockDim.x*blockIdx.x+threadIdx.x;
    if(index>=n*d) return;
    int row=index/d,column=index%d;
    double positive=0.0;
    for(int sign=0;sign<2;++sign) {
        int probe=2*column+sign;
        double x,y,z;
        if(kind==0 || kind==1) {
            x=source[3*row]+weighted(weights,parameters,row,probe,controls,width,0,3,compensated);
            y=source[3*row+1]+weighted(weights,parameters,row,probe,controls,width,1,3,compensated);
            z=source[3*row+2]+weighted(weights,parameters,row,probe,controls,width,2,3,compensated);
        } else {
            double depth;
            if(kind==2) depth=parameters[probe*width]*depths[row]+parameters[probe*width+1];
            else {
                double nominal=depths[row];
                if(kind==4) nominal=nominal+parameters[probe*width];
                depth=nominal+weighted(weights,parameters,row,probe,controls,width,kind==4?1:0,1,compensated);
            }
            x=source[3*row]*depth;y=source[3*row+1]*depth;z=source[3*row+2]*depth;
        }
        double a[3],b[3];transformed(x,y,z,first,a);
        if(kind==1) {
            x=target[3*row]+weighted(target_weights,parameters,row,probe,controls,width,0,3,compensated);
            y=target[3*row+1]+weighted(target_weights,parameters,row,probe,controls,width,1,3,compensated);
            z=target[3*row+2]+weighted(target_weights,parameters,row,probe,controls,width,2,3,compensated);
            transformed(x,y,z,second,b);
        } else {b[0]=planes[3*row];b[1]=planes[3*row+1];b[2]=planes[3*row+2];}
        double dx=a[0]-b[0],dy=a[1]-b[1],dz=a[2]-b[2];
        double residual=sqrt_weights[row]*(
            normals[3*row]*dx+normals[3*row+1]*dy+normals[3*row+2]*dz);
        if(!isfinite(residual)) atomicExch(invalid,1);
        if(sign==0) positive=residual;
        else output[index]=(positive-residual)/(2.0*steps[column]);
    }
}
extern "C" __global__ void poses(int pair,int n,int d,const double* points,
 const double* target,const double* planes,const double* normals,const double* sqrt_weights,
 const double* first,const double* second,const double* steps,double* output,int* invalid) {
    int index=blockDim.x*blockIdx.x+threadIdx.x;
    if(index>=n*d) return;
    int row=index/d,column=index%d;
    double positive=0.0;
    for(int sign=0;sign<2;++sign) {
        int probe=2*column+sign;
        double a[3],b[3];
        transformed(points[3*row],points[3*row+1],points[3*row+2],first+11*probe,a);
        if(pair) transformed(target[3*row],target[3*row+1],target[3*row+2],second+11*probe,b);
        else {b[0]=planes[3*row];b[1]=planes[3*row+1];b[2]=planes[3*row+2];}
        double dx=a[0]-b[0],dy=a[1]-b[1],dz=a[2]-b[2];
        double residual=sqrt_weights[row]*(
            normals[3*row]*dx+normals[3*row+1]*dy+normals[3*row+2]*dz);
        if(!isfinite(residual)) atomicExch(invalid,1);
        if(sign==0) positive=residual;
        else output[index]=(positive-residual)/(2.0*steps[column]);
    }
}

extern "C" __global__ void calibrate(int kind,int n,int controls,int width,const double* source,
 const double* depths,const double* weights,const double* parameters,
 int compensated,double* output) {
    int row=blockDim.x*blockIdx.x+threadIdx.x;
    if(row>=n) return;
    if(kind==0 || kind==1) {
        for(int axis=0;axis<3;++axis)
            output[3*row+axis]=source[3*row+axis]+weighted(weights,parameters,row,0,controls,width,axis,3,compensated);
    } else {
        double depth;
        if(kind==2) depth=parameters[0]*depths[row]+parameters[1];
        else {
            double nominal=depths[row];
            if(kind==4) nominal=nominal+parameters[0];
            depth=nominal+weighted(weights,parameters,row,0,controls,width,kind==4?1:0,1,compensated);
        }
        for(int axis=0;axis<3;++axis) output[3*row+axis]=source[3*row+axis]*depth;
    }
}
extern "C" __global__ void evaluate(int pair,int n,const double* points,const double* target,
 const double* planes,const double* normals,const double* sqrt_weights,const double* first,
 const double* second,double* output) {
    int row=blockDim.x*blockIdx.x+threadIdx.x;
    if(row>=n) return;
    double a[3],b[3];transformed(points[3*row],points[3*row+1],points[3*row+2],first,a);
    if(pair) transformed(target[3*row],target[3*row+1],target[3*row+2],second,b);
    else {b[0]=planes[3*row];b[1]=planes[3*row+1];b[2]=planes[3*row+2];}
    double dx=a[0]-b[0],dy=a[1]-b[1],dz=a[2]-b[2];
    output[row]=sqrt_weights[row]*(normals[3*row]*dx+normals[3*row+1]*dy+normals[3*row+2]*dz);
}

"""
