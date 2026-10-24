"use strict";
(() => {
  function transform(p,tf){
    const [x,y,z,w]=tf.rotation_quat_xyzw,t=tf.translation_m;
    const u=[2*(y*p[2]-z*p[1]),2*(z*p[0]-x*p[2]),2*(x*p[1]-y*p[0])];
    return [p[0]+w*u[0]+y*u[2]-z*u[1]+t[0],p[1]+w*u[1]+z*u[0]-x*u[2]+t[1],p[2]+w*u[2]+x*u[1]-y*u[0]+t[2]];
  }
  function move(p,s){
    const [a,b,c]=s,ca=Math.cos(a),sa=Math.sin(a),cb=Math.cos(b),sb=Math.sin(b),cc=Math.cos(c),sc=Math.sin(c);
    const y=ca*p[1]-sa*p[2],z=sa*p[1]+ca*p[2],x=cb*p[0]+sb*z;
    return [cc*x-sc*y+s[3],sc*x+cc*y+s[4],-sb*p[0]+cb*z+s[5]];
  }
  function solveLinear(matrix,rhs){
    const n=rhs.length,a=matrix.map((row,i)=>[...row,rhs[i]]);
    for(let k=0;k<n;k++){
      let pivot=k;for(let i=k+1;i<n;i++)if(Math.abs(a[i][k])>Math.abs(a[pivot][k]))pivot=i;
      if(Math.abs(a[pivot][k])<1e-14)throw Error("Degenerate live fit geometry");
      [a[k],a[pivot]]=[a[pivot],a[k]];const v=a[k][k];for(let j=k;j<=n;j++)a[k][j]/=v;
      for(let i=0;i<n;i++)if(i!==k){const f=a[i][k];for(let j=k;j<=n;j++)a[i][j]-=f*a[k][j];}
    }
    return a.map(row=>row[n]);
  }
  function geometry(points){
    const n=points.length,mean=[0,1,2].map(k=>points.reduce((s,p)=>s+p[k],0)/n);
    const a=Array.from({length:3},()=>Array(3).fill(0));
    for(const p of points){const q=p.map((v,k)=>v-mean[k]),r=q.reduce((s,v)=>s+v*v,0);for(let i=0;i<3;i++)for(let j=0;j<3;j++)a[i][j]+=(i===j?r:0)-q[i]*q[j];}
    // Jacobi eigenvalues of the centred rotational information block. No damping
    // enters this diagnostic: damping cannot make an unobservable axis measured.
    for(let step=0;step<32;step++){
      let p=0,q=1;for(let i=0;i<3;i++)for(let j=i+1;j<3;j++)if(Math.abs(a[i][j])>Math.abs(a[p][q])){p=i;q=j;}
      if(Math.abs(a[p][q])<1e-14)break;
      const angle=.5*Math.atan2(2*a[p][q],a[q][q]-a[p][p]),c=Math.cos(angle),s=Math.sin(angle);
      const pp=a[p][p],qq=a[q][q],pq=a[p][q];
      for(let k=0;k<3;k++)if(k!==p&&k!==q){const kp=a[k][p],kq=a[k][q];a[k][p]=a[p][k]=c*kp-s*kq;a[k][q]=a[q][k]=s*kp+c*kq;}
      a[p][p]=c*c*pp-2*s*c*pq+s*s*qq;a[q][q]=s*s*pp+2*s*c*pq+c*c*qq;a[p][q]=a[q][p]=0;
    }
    const eigenvalues=[a[0][0],a[1][1],a[2][2]].map(v=>Math.max(0,v)).sort((x,y)=>x-y),largest=eigenvalues[2];
    return {rank:3+eigenvalues.filter(v=>v>Math.max(1e-14,largest*1e-8)).length,rotationRatio:largest?eigenvalues[0]/largest:0};
  }
  class LiveFit {
    constructor(problem,options={}){
      this.scenario=options.scenario||"normal";this.loss=options.loss||"least_squares";this.huberDelta=.05;
      if(!["normal","limited","outliers"].includes(this.scenario)||!["least_squares","huber"].includes(this.loss))throw Error("Unknown live fit scenario or loss");
      this.problem=problem;this.base=problem.source.map(p=>transform(p,problem.initial_transform));
      this.fit=new Set(problem.fit_ids);this.fitIndices=problem.ids.map((id,i)=>this.fit.has(id)?i:-1).filter(i=>i>=0);
      this.holdIndices=problem.ids.map((id,i)=>this.fit.has(id)?-1:i).filter(i=>i>=0);
      if(this.fitIndices.length<3||!this.holdIndices.length)throw Error("Invalid fit/holdout split");
      this.targets=problem.target.map(p=>[...p]);this.injectedIndices=[];
      if(this.scenario==="limited"){
        let pair,distance=Infinity;
        for(const i of this.fitIndices)for(const j of this.fitIndices)if(j>i){const d=Math.hypot(...this.base[i].map((v,k)=>v-this.base[j][k]));if(d>1e-8&&d<distance){distance=d;pair=[i,j];}}
        if(!pair)throw Error("No distinct measured capture pair");this.fitIndices=pair;
      }else if(this.scenario==="outliers"){
        this.injectedIndices=this.fitIndices.filter((_,i)=>i%8===0);
        for(const i of this.injectedIndices)this.targets[i]=this.targets[i].map((v,k)=>v+[.6,-.4,.5][k]);
      }
      this.geometry=geometry(this.fitIndices.map(i=>this.base[i]));
      this.state=[.025,.05,-.02,.05,-.03,.02];this.iteration=0;this.status="ready";this.history=[];this.record();
    }
    residuals(state,indices,targets=this.targets){return indices.flatMap(i=>move(this.base[i],state).map((v,k)=>v-targets[i][k]));}
    objective(state){
      const r=this.residuals(state,this.fitIndices);let total=0;
      for(let i=0;i<r.length;i+=3){const norm=Math.hypot(r[i],r[i+1],r[i+2]);total+=this.loss==="huber"&&norm>this.huberDelta?2*this.huberDelta*norm-this.huberDelta**2:norm*norm;}
      return total;
    }
    record(){
      const rmse=ids=>Math.sqrt(this.residuals(this.state,ids).reduce((s,v)=>s+v*v,0)/ids.length);
      this.history.push({iteration:this.iteration,fit:rmse(this.fitIndices),holdout:rmse(this.holdIndices),objective:this.objective(this.state)});
    }
    step(){
      if(["converged","stalled","iteration_limit","underconstrained"].includes(this.status))return false;
      if(this.geometry.rank<6){this.status="underconstrained";return false;}
      const r=this.residuals(this.state,this.fitIndices),jac=[];
      for(let k=0;k<6;k++){const s=[...this.state];s[k]+=1e-5;const v=this.residuals(s,this.fitIndices);jac.push(v.map((x,i)=>(x-r[i])/1e-5));}
      const weights=r.map((_,i)=>{const start=i-i%3,norm=Math.hypot(r[start],r[start+1],r[start+2]);return this.loss==="huber"&&norm>this.huberDelta?this.huberDelta/norm:1;});
      const h=Array.from({length:6},()=>Array(6).fill(0)),g=Array(6).fill(0);
      for(let k=0;k<6;k++){for(let i=0;i<r.length;i++)g[k]-=weights[i]*jac[k][i]*r[i];for(let j=0;j<6;j++)h[k][j]=jac[k].reduce((s,v,i)=>s+weights[i]*v*jac[j][i],0);h[k][k]+=1e-5;}
      const delta=solveLinear(h,g),before=this.objective(this.state);let scale=1,next;
      for(let i=0;i<16;i++){next=this.state.map((v,k)=>v+scale*delta[k]);if(this.objective(next)<=before)break;scale*=.5;next=null;}
      if(!next){this.status="stalled";return false;}
      this.state=next;this.iteration++;this.status="running";this.record();
      if(Math.hypot(...delta)*scale<1e-8||before-this.objective(next)<1e-12)this.status="converged";
      else if(this.iteration>=40)this.status="iteration_limit";
      return this.status==="running";
    }
    points(){return this.base.map(p=>move(p,this.state));}
  }
  globalThis.CalibrexDemoSolver={transform,move,LiveFit,solveLinear,geometry};
})();
