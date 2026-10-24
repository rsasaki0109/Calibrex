// A viewer for two saved real-data point clouds. No solver or interpolated optimizer states.
"use strict";
(() => {
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  function validate(data) {
    if (data?.schema !== "calibrex.browser_alignment_demo/v1") throw Error("Unsupported sample data");
    for (const key of ["reference", "before", "after"]) {
      const values = data[key];
      if (!Array.isArray(values) || !values.length || values.length % 3 || values.length > 300000 || !values.every(Number.isFinite)) throw Error("Invalid sample points");
    }
    if (data.before.length !== data.after.length || !Array.isArray(data.camera?.target) || data.camera.target.length !== 3 || !data.camera.target.every(Number.isFinite) ||
        ![data.camera.azimuth_deg, data.camera.elevation_deg, data.camera.distance_m].every(Number.isFinite) || data.camera.distance_m <= 0) throw Error("Invalid sample camera");
    if (![data.metrics?.start_residual_m, data.metrics?.estimate_residual_m].every((v) => Number.isFinite(v) && v >= 0)) throw Error("Invalid sample metrics");
    return data;
  }
  function basis(camera) {
    const az = camera.azimuth_deg * Math.PI / 180, el = camera.elevation_deg * Math.PI / 180;
    const forward = [-Math.cos(el)*Math.cos(az), -Math.cos(el)*Math.sin(az), -Math.sin(el)];
    const right = [-Math.sin(az), Math.cos(az), 0];
    const up = [-Math.sin(el)*Math.cos(az), -Math.sin(el)*Math.sin(az), Math.cos(el)];
    const eye = camera.target.map((v,i) => v-forward[i]*camera.distance_m);
    return {eye, forward, right, up};
  }
  function project(points, camera, width, height, layer) {
    const {eye,forward,right,up} = basis(camera), focal = Math.min(width*.85,height*1.35), out=[];
    for(let i=0;i<points.length;i+=3) {
      const x=points[i]-eye[0], y=points[i+1]-eye[1], z=points[i+2]-eye[2];
      const depth=x*forward[0]+y*forward[1]+z*forward[2];
      if(depth<.3) continue;
      const u=width/2+focal*(x*right[0]+y*right[1]+z*right[2])/depth;
      const v=height/2-focal*(x*up[0]+y*up[1]+z*up[2])/depth;
      if(u>=-3 && u<=width+3 && v>=-3 && v<=height+3) out.push({u,v,depth,layer});
    }
    return out;
  }
  class AlignmentDemo {
    constructor(host) {
      this.host=host; this.canvas=host.querySelector("canvas"); this.ctx=this.canvas.getContext("2d");
      this.slider=host.querySelector("input[type=range]"); this.split=50; this.cache=null;
      this.slider.addEventListener("input",()=>{this.stop();this.setSplit(Number(this.slider.value));});
      host.querySelectorAll("[data-compare]").forEach((button)=>button.addEventListener("click",()=>{this.stop();this.setSplit(Number(button.dataset.compare));}));
      host.querySelector("[data-reveal]").addEventListener("click",()=>{if(this.playing)this.stop();else this.reveal();});
      host.querySelectorAll("[data-cloud-zoom]").forEach((button)=>button.addEventListener("click",()=>{
        if(!this.camera) return;
        this.camera.distance_m=clamp(this.camera.distance_m*(button.dataset.cloudZoom==="in"?.8:1.25),5,60);this.invalidate();
      }));
      host.querySelector("[data-cloud-reset]").addEventListener("click",()=>this.reset());
      this.canvas.addEventListener("pointerdown",(e)=>{if(e.button!==0 || !this.camera)return;this.stop();this.drag={x:e.clientX,y:e.clientY};this.canvas.setPointerCapture(e.pointerId);this.canvas.focus({preventScroll:true});});
      this.canvas.addEventListener("pointermove",(e)=>{
        if(!this.drag)return;
        this.camera.azimuth_deg-=(e.clientX-this.drag.x)*.3;
        this.camera.elevation_deg=clamp(this.camera.elevation_deg+(e.clientY-this.drag.y)*.2,5,80);
        this.drag={x:e.clientX,y:e.clientY};this.invalidate();
      });
      for(const name of ["pointerup","pointercancel","lostpointercapture"])this.canvas.addEventListener(name,()=>{this.drag=null;});
      this.canvas.addEventListener("keydown",(e)=>{
        if(!this.camera || !["ArrowLeft","ArrowRight","ArrowUp","ArrowDown","+","=","-","Home"].includes(e.key))return;
        e.preventDefault();this.stop();
        if(e.key==="Home"){this.reset();return;}
        this.camera.azimuth_deg+=e.key==="ArrowLeft"?-5:e.key==="ArrowRight"?5:0;
        this.camera.elevation_deg=clamp(this.camera.elevation_deg+(e.key==="ArrowUp"?5:e.key==="ArrowDown"?-5:0),5,80);
        if(["+","=","-"].includes(e.key))this.camera.distance_m=clamp(this.camera.distance_m*(e.key==="-"?1.25:.8),5,60);
        this.invalidate();
      });
      this.observer=new ResizeObserver(()=>this.invalidate());this.observer.observe(this.canvas);
      this.setSplit(50);
    }
    async load() {
      if(this.data)return;
      if(this.loading)return this.loading;
      this.host.querySelector(".cloud-state").textContent="Loading real point clouds…";
      this.loading=(async()=>{
        try {
          const response=await fetch("samples/alignment_demo.json");
          if(!response.ok)throw Error(`Sample download failed (${response.status})`);
          this.data=validate(await response.json());
          this.host.querySelector(".cloud-before-value").textContent=(this.data.metrics.start_residual_m*100).toFixed(1);
          this.host.querySelector(".cloud-after-value").textContent=(this.data.metrics.estimate_residual_m*100).toFixed(1);
          this.host.querySelector(".cloud-point-count").textContent=new Intl.NumberFormat().format((this.data.reference.length+this.data.after.length)/3);
          this.host.querySelector(".cloud-state").textContent="REAL RECORDING / SAVED COMPARISON";
          this.host.querySelector(".cloud-error").hidden=true;
          this.reset();
        } catch(error) {
          this.host.querySelector(".cloud-state").textContent="Sample unavailable";
          const box=this.host.querySelector(".cloud-error");box.hidden=false;box.textContent=error.message+". Select Explore sample to retry.";
          throw error;
        } finally {this.loading=null;}
      })();
      return this.loading;
    }
    reset(){if(!this.data)return;this.stop();this.camera=structuredClone(this.data.camera);this.invalidate();}
    setSplit(value){
      this.split=clamp(value,0,100);this.slider.value=this.split;
      this.host.querySelector(".cloud-divider").style.left=this.split+"%";
      this.host.querySelector(".cloud-divider").hidden=this.split===0||this.split===100;
      this.host.querySelector(".cloud-before-label").hidden=this.split===0;
      this.host.querySelector(".cloud-after-label").hidden=this.split===100;
      this.host.querySelectorAll("[data-compare]").forEach((button)=>button.setAttribute("aria-pressed",String(Number(button.dataset.compare)===this.split)));
      this.draw();
    }
    invalidate(){this.cache=null;if(!this.pending)this.pending=requestAnimationFrame(()=>{this.pending=null;this.draw();});}
    makeFrame(target,width,height,dpr){
      const canvas=document.createElement("canvas");canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
      const ctx=canvas.getContext("2d");ctx.scale(dpr,dpr);
      const glow=ctx.createRadialGradient(width*.5,height*.45,0,width*.5,height*.5,width*.7);
      glow.addColorStop(0,"#122436");glow.addColorStop(1,"#050b13");ctx.fillStyle=glow;ctx.fillRect(0,0,width,height);
      const points=[...project(this.data.reference,this.camera,width,height,0),...project(target,this.camera,width,height,1)].sort((a,b)=>b.depth-a.depth);
      for(const p of points){
        ctx.fillStyle=p.layer?"#ffb15b":"#48dbef";
        ctx.globalAlpha=clamp(.95-p.depth/100,.3,.85);
        const size=clamp(32/p.depth,1.15,2.8);ctx.fillRect(p.u,p.v,size,size);
      }
      ctx.globalAlpha=1;return canvas;
    }
    draw(){
      if(!this.data||!this.camera||!this.ctx)return;
      const width=this.canvas.clientWidth,height=this.canvas.clientHeight;if(!width||!height)return;
      const dpr=Math.min(devicePixelRatio||1,1.5);
      if(!this.cache){
        this.canvas.width=Math.round(width*dpr);this.canvas.height=Math.round(height*dpr);
        this.cache={before:this.makeFrame(this.data.before,width,height,dpr),after:this.makeFrame(this.data.after,width,height,dpr)};
      }
      const w=this.canvas.width,h=this.canvas.height,cut=Math.round(w*this.split/100),ctx=this.ctx;
      ctx.clearRect(0,0,w,h);
      if(cut)ctx.drawImage(this.cache.before,0,0,cut,h,0,0,cut,h);
      if(cut<w)ctx.drawImage(this.cache.after,cut,0,w-cut,h,cut,0,w-cut,h);
    }
    stop(){if(this.playing)cancelAnimationFrame(this.playing);this.playing=null;this.host.querySelector("[data-reveal]").textContent="Play reveal";}
    reveal(){
      this.stop();if(!this.data)return;
      if(matchMedia("(prefers-reduced-motion: reduce)").matches){this.setSplit(50);return;}
      this.host.querySelector("[data-reveal]").textContent="Pause reveal";
      const start=performance.now();
      const animate=(now)=>{const t=Math.min(1,(now-start)/4000);this.setSplit(50+47*Math.sin(t*2*Math.PI));if(t<1)this.playing=requestAnimationFrame(animate);else this.stop();};
      this.playing=requestAnimationFrame(animate);
    }
  }
  globalThis.CalibrexAlignment={validate,basis,project,AlignmentDemo};
})();
