"use strict";
(() => {
  const WIDTH=960,HEIGHT=520;
  const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
  function unproject(data){
    const [w,h]=data.image_size||[],f=data.focal_px;
    if(!Number.isInteger(w)||!Number.isInteger(h)||w<=0||h<=0||!Number.isFinite(f)||f<=0||!Array.isArray(data.correspondences)||!data.correspondences.length)throw Error("Invalid stereo cloud");
    return data.correspondences.map((p,id)=>{
      const z=p.depth_m,[u,v]=p.left||[];
      if(![u,v,z].every(Number.isFinite)||z<=0||u<0||u>=w||v<0||v>=h)throw Error("Invalid stereo cloud point");
      return {id,xyz:[(u-(w-1)/2)*z/f,(v-(h-1)/2)*z/f,z],z};
    });
  }
  function bounds(points,limit=Infinity){
    const included=points.filter(p=>p.z<=limit);
    const lo=[0,0,0],hi=[0,0,0];
    for(const p of included)for(let k=0;k<3;k++){lo[k]=Math.min(lo[k],p.xyz[k]);hi[k]=Math.max(hi[k],p.xyz[k]);}
    const center=lo.map((x,k)=>(x+hi[k])/2);
    const radius=Math.max(.05,Math.hypot(...hi.map((x,k)=>(x-lo[k])/2)));
    return {center,radius};
  }
  function project(xyz,view){
    const [x,y,z]=xyz.map((v,k)=>v-view.center[k]),a=view.yaw*Math.PI/180,b=view.pitch*Math.PI/180;
    const rx=Math.cos(a)*x+Math.sin(a)*z,rz=-Math.sin(a)*x+Math.cos(a)*z;
    const ry=Math.cos(b)*y-Math.sin(b)*rz,depth=Math.sin(b)*y+Math.cos(b)*rz,scale=205/view.radius*view.zoom;
    return {x:WIDTH/2+rx*scale,y:HEIGHT/2+ry*scale,depth};
  }
  function pick(points,x,y,radius){
    let selected=null,best=radius*radius;
    for(const p of points){const d=(p.x-x)**2+(p.y-y)**2;if(d<best-1e-9||(Math.abs(d-best)<1e-9&&(!selected||p.depth<selected.depth))){selected=p;best=d;}}
    return selected?.id??null;
  }
  function colour(z){const a=clamp((z-.3)/2.7,0,1);return `rgb(${Math.floor(255*(1-a))} ${Math.floor(130+100*a)} ${Math.floor(80+175*a)})`;}
  class StereoCloud {
    constructor(root,onSelect){
      this.root=root;this.onSelect=onSelect;this.points=[];this.selected=0;this.limit=1;this.projected=[];
      root.innerHTML=`<div class="stereo-cloud-heading"><div><span class="eyebrow">MEASURED MATCHES / ESTIMATED 3D</span><h3>Give the depth some perspective.</h3></div><label>Display depth range <select aria-label="Stereo 3D depth range"><option value="1">Z ≤ 1 m</option><option value="3">Z ≤ 3 m</option><option value="all">All estimated depths</option></select></label></div><canvas width="960" height="520" role="img" aria-label="Estimated stereo point cloud" class="stereo-cloud-canvas"></canvas><div class="stereo-cloud-controls"><label>Orbit azimuth <input type="range" min="-180" max="180" step="1" value="-55" aria-label="Stereo 3D orbit azimuth"></label><label>Orbit elevation <input type="range" min="-75" max="75" step="1" value="20" aria-label="Stereo 3D orbit elevation"></label><label>View zoom <input type="range" min="0.6" max="4" step="0.1" value="1" aria-label="Stereo 3D zoom"></label><button type="button" data-cloud-reset>Reset 3D view</button><button type="button" data-cloud-focus>Focus selected match</button></div><p class="stereo-cloud-view"></p><p class="stereo-cloud-summary" role="status"></p><output class="stereo-cloud-position"></output><p class="stereo-cloud-note">Drag to orbit; click a point to inspect its image match. Sliders and the match controls provide keyboard access. Coordinates: rectified left camera, X right · Y down · Z forward, metres. Orange ≤ 0.3 m → cyan ≥ 3 m; only colours clip. This is sparse estimated depth, with possible wrong matches and no independent depth truth.</p>`;
      this.canvas=root.querySelector("canvas");
      this.resize=new ResizeObserver(()=>{if(this.view)this.draw();});this.resize.observe(this.canvas);
      root.querySelector("select").addEventListener("change",e=>{this.limit=e.target.value==="all"?Infinity:Number(e.target.value);this.fit(false);});
      for(const [key,label] of [["yaw","azimuth"],["pitch","elevation"],["zoom","zoom"]]){
        const input=root.querySelector(`[aria-label="Stereo 3D ${key==="zoom"?"zoom":"orbit "+label}"]`);
        input.addEventListener("input",()=>{if(!this.view)return;this.view[key]=Number(input.value);this.draw();});
      }
      root.querySelector("[data-cloud-reset]").addEventListener("click",()=>this.fit(true));
      root.querySelector("[data-cloud-focus]").addEventListener("click",()=>{
        const p=this.points[this.selected];if(!p)return;
        if(p.z>this.limit){this.limit=Infinity;root.querySelector("select").value="all";this.fit(false);}
        this.view.center=p.xyz.slice();this.view.zoom=3;this.sync();this.draw();
      });
      this.canvas.addEventListener("pointerdown",e=>{if(!this.view||e.button!==0)return;this.drag={x:e.clientX,y:e.clientY,yaw:this.view.yaw,pitch:this.view.pitch,moved:false};this.canvas.setPointerCapture(e.pointerId);});
      this.canvas.addEventListener("pointermove",e=>{
        const d=this.drag;if(!d)return;const dx=e.clientX-d.x,dy=e.clientY-d.y;
        if(Math.hypot(dx,dy)>4)d.moved=true;if(!d.moved)return;
        this.view.yaw=((d.yaw+dx*.4+540)%360+360)%360-180;this.view.pitch=clamp(d.pitch-dy*.3,-75,75);this.sync();this.draw();
      });
      this.canvas.addEventListener("pointerup",e=>{
        const d=this.drag;this.drag=null;if(!d||d.moved)return;
        const box=this.canvas.getBoundingClientRect(),id=pick(this.projected,(e.clientX-box.left)/box.width*WIDTH,(e.clientY-box.top)/box.height*HEIGHT,12/box.width*WIDTH);
        if(id!==null)this.onSelect(id);
      });
      for(const event of ["pointercancel","lostpointercapture"])this.canvas.addEventListener(event,()=>{this.drag=null;});
    }
    clear(){this.points=[];this.projected=[];this.view=null;this.drag=null;}
    setData(data){this.points=unproject(data);this.selected=0;this.limit=1;this.root.querySelector("select").value="1";this.fit(true);}
    select(id){if(!this.points[id])return;this.selected=id;this.draw();}
    fit(reset){
      if(!this.points.length)return;
      const previous=this.view;this.view={...bounds(this.points,this.limit),yaw:reset?-55:(previous?.yaw??-55),pitch:reset?20:(previous?.pitch??20),zoom:1};this.sync();this.draw();
    }
    sync(){
      for(const [name,value] of [["orbit azimuth",this.view.yaw],["orbit elevation",this.view.pitch],["zoom",this.view.zoom]])this.root.querySelector(`[aria-label="Stereo 3D ${name}"]`).value=String(value);
    }
    draw(){
      if(!this.view)return;
      const ctx=this.canvas.getContext("2d"),view=this.view,unit=WIDTH/(this.canvas.getBoundingClientRect().width||WIDTH);
      const bg=ctx.createRadialGradient(470,210,10,480,260,550);bg.addColorStop(0,"#18354b");bg.addColorStop(1,"#07131f");ctx.fillStyle=bg;ctx.fillRect(0,0,WIDTH,HEIGHT);
      const origin=project([0,0,0],view),length=view.radius*.45;
      ctx.font=`${12*unit}px monospace`;
      let step=10**Math.floor(Math.log10(view.radius));if(view.radius/step>=5)step*=2;
      ctx.strokeStyle="#6ea3b92c";ctx.lineWidth=unit;
      const line=(a,b)=>{const p=project(a,view),q=project(b,view);ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.stroke();};
      for(let i=-4;i<=4;i++)line([i*step,0,0],[i*step,0,8*step]);
      for(let i=0;i<=8;i++)line([-4*step,0,i*step],[4*step,0,i*step]);
      for(const [axis,tint,label] of [[0,"#ffad89","X"],[1,"#89e3b8","Y"],[2,"#9cbbff","Z"]]){
        const v=[0,0,0];v[axis]=length;const end=project(v,view);ctx.strokeStyle=tint;ctx.lineWidth=1.5*unit;ctx.beginPath();ctx.moveTo(origin.x,origin.y);ctx.lineTo(end.x,end.y);ctx.stroke();ctx.fillStyle=tint;ctx.fillText(label,end.x+5,end.y-5);
      }
      this.projected=this.points.filter(p=>p.z<=this.limit).map(p=>({...project(p.xyz,view),id:p.id,z:p.z})).sort((a,b)=>b.depth-a.depth);
      for(const p of this.projected){ctx.fillStyle=colour(p.z);ctx.globalAlpha=.8;ctx.beginPath();ctx.arc(p.x,p.y,3*unit,0,Math.PI*2);ctx.fill();}ctx.globalAlpha=1;
      const selected=this.points[this.selected],visible=selected.z<=this.limit;
      if(visible){const p=project(selected.xyz,view);ctx.strokeStyle="#d8edff66";ctx.setLineDash([4,5]);ctx.beginPath();ctx.moveTo(origin.x,origin.y);ctx.lineTo(p.x,p.y);ctx.stroke();ctx.setLineDash([]);ctx.strokeStyle="#fff";ctx.lineWidth=2*unit;ctx.beginPath();ctx.arc(p.x,p.y,8*unit,0,Math.PI*2);ctx.stroke();ctx.fillStyle="#fff";ctx.fillText("#"+(selected.id+1),p.x+12*unit,p.y-12*unit);}
      ctx.fillStyle="#abc7da";ctx.fillRect(origin.x-3*unit,origin.y-3*unit,6*unit,6*unit);
      this.root.querySelector(".stereo-cloud-view").textContent=`Azimuth ${view.yaw.toFixed(0)}° · elevation ${view.pitch.toFixed(0)}° · ${view.zoom.toFixed(1)}× view zoom · grid ${Number(step.toPrecision(2))} m on camera Y=0 plane`;
      const outside=this.points.length-this.projected.length;
      this.root.querySelector(".stereo-cloud-summary").textContent=`${this.projected.length} / ${this.points.length} points in depth range · ${outside} ${this.limit===Infinity?"outside depth range":"beyond "+this.limit+" m"}. ${visible?"White ring: selected match.":"Selected match is outside this range; use Focus selected match to reveal it."}`;
      this.root.querySelector(".stereo-cloud-position").textContent=`Match ${selected.id+1} · X ${selected.xyz[0].toFixed(3)} m · Y ${selected.xyz[1].toFixed(3)} m · Z ${selected.xyz[2].toFixed(3)} m`;
      this.canvas.setAttribute("aria-label",`Estimated stereo point cloud. ${this.projected.length} of ${this.points.length} points in depth range; ${outside} outside. Selected match ${selected.id+1}${visible?"":" outside range"}. Use orbit sliders to rotate and match controls to select. No independent depth truth.`);
    }
  }
  globalThis.CalibrexStereoCloud={unproject,bounds,project,pick,colour,StereoCloud};
})();
