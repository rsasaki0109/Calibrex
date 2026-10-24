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
  class LiveFit {
    constructor(problem){
      this.problem=problem;this.base=problem.source.map(p=>transform(p,problem.initial_transform));
      this.fit=new Set(problem.fit_ids);this.fitIndices=problem.ids.map((id,i)=>this.fit.has(id)?i:-1).filter(i=>i>=0);
      this.holdIndices=problem.ids.map((id,i)=>this.fit.has(id)?-1:i).filter(i=>i>=0);
      if(this.fitIndices.length<3||!this.holdIndices.length)throw Error("Invalid fit/holdout split");
      this.state=[.025,.05,-.02,.05,-.03,.02];this.iteration=0;this.status="ready";this.history=[];this.record();
    }
    residuals(state,indices){return indices.flatMap(i=>move(this.base[i],state).map((v,k)=>v-this.problem.target[i][k]));}
    objective(state){return this.residuals(state,this.fitIndices).reduce((s,v)=>s+v*v,0);}
    record(){
      const rmse=ids=>Math.sqrt(this.residuals(this.state,ids).reduce((s,v)=>s+v*v,0)/ids.length);
      this.history.push({iteration:this.iteration,fit:rmse(this.fitIndices),holdout:rmse(this.holdIndices)});
    }
    step(){
      if(["converged","stalled","iteration_limit"].includes(this.status))return false;
      const r=this.residuals(this.state,this.fitIndices),jac=[];
      for(let k=0;k<6;k++){const s=[...this.state];s[k]+=1e-5;const v=this.residuals(s,this.fitIndices);jac.push(v.map((x,i)=>(x-r[i])/1e-5));}
      const h=Array.from({length:6},()=>Array(6).fill(0)),g=Array(6).fill(0);
      for(let k=0;k<6;k++){for(let i=0;i<r.length;i++)g[k]-=jac[k][i]*r[i];for(let j=0;j<6;j++)h[k][j]=jac[k].reduce((s,v,i)=>s+v*jac[j][i],0);h[k][k]+=1e-5;}
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
  globalThis.CalibrexDemoSolver={transform,move,LiveFit,solveLinear};
})();
