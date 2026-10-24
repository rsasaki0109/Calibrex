"use strict";
(() => {
  const AXES=["Roll","Pitch","Yaw","X","Y","Z"];
  function validate(data){
    if(!data||data.schema!=="calibrex.camera_lidar_projection_demo/v1"||data.capture_kind!=="real"||![60,90].includes(data.frame)||!Array.isArray(data.image_size)||data.image_size.length!==2||!data.image_size.every(v=>Number.isInteger(v)&&v>0&&v<=4096)||!Array.isArray(data.intrinsics)||data.intrinsics.length!==4||!data.intrinsics.every(Number.isFinite)||data.intrinsics[0]<=0||data.intrinsics[1]<=0||!Array.isArray(data.points)||data.points.length<1||data.points.length>15000||!data.points.every(p=>Array.isArray(p)&&p.length===3&&p.every(v=>Number.isFinite(v)&&Math.abs(v)<1000)))throw Error("Invalid camera-LiDAR demo data");
    return data;
  }
  function project(point,intrinsics,offsets){
    const [rx,ry,rz]=offsets.slice(0,3).map(v=>v*Math.PI/180);
    const [sx,sy,sz]=[Math.sin(rx),Math.sin(ry),Math.sin(rz)];
    const [cx,cy,cz]=[Math.cos(rx),Math.cos(ry),Math.cos(rz)];
    const [x,y,z]=point;
    const y1=cx*y-sx*z,z1=sx*y+cx*z;
    const x2=cy*x+sy*z1,z2=-sy*x+cy*z1;
    const qx=cz*x2-sz*y1+offsets[3],qy=sz*x2+cz*y1+offsets[4],qz=z2+offsets[5];
    if(qz<=.01)return null;
    return [intrinsics[0]*qx/qz+intrinsics[2],intrinsics[1]*qy/qz+intrinsics[3],qz];
  }
  class CameraLidarDemo {
    constructor(panel,json){
      this.panel=panel;this.json=json;this.sequence=0;this.offsets=[0,3,0,0,0,0];this.split=50;
      panel.innerHTML=`<div class="method-media"><div class="cloud-topline"><span>REAL KITTI / INTERACTIVE PROJECTION</span><span>Colour = camera depth · 5–40 m</span></div><div class="projection-toolbar"><label>Scene <select aria-label="Camera-LiDAR scene"><option value="60">Frame 0060</option><option value="90">Frame 0090</option></select></label><span>Same real photograph and LiDAR returns</span></div><div class="projection-stage"><canvas width="1242" height="375" role="img" aria-label="Real KITTI image with vendor and adjusted LiDAR projections"></canvas><span class="projection-label projection-vendor">VENDOR</span><span class="projection-label projection-adjusted">YOUR ADJUSTMENT</span></div><div class="projection-content" hidden><div class="projection-compare"><label for="projection-split">Compare overlays</label><input id="projection-split" aria-label="Vendor and adjusted overlay split" type="range" min="0" max="100" value="50"><output>50% vendor</output></div><div class="cloud-controls"><button type="button" data-preset="perturbed">Inject +3° pitch</button><button type="button" data-preset="vendor">Restore vendor</button></div><div class="projection-sliders"></div><details class="projection-translation"><summary>Translation offsets</summary><div class="projection-translation-sliders"></div></details><div class="projection-readouts" aria-live="polite"></div><p class="projection-help">The divider reveals the vendor projection on the left and your adjustment on the right. Offsets act in the rectified camera frame. This changes the overlay, not the calibration file.</p></div><p class="method-status" role="status">Loading real image and points…</p><button class="projection-retry" type="button" hidden>Retry scene</button></div><aside class="cloud-evidence"><div class="eyebrow">CAMERA ↔ LIDAR / TRY THE ALIGNMENT</div><h2>Move the points.<br>See the mismatch.</h2><p class="cloud-description">Slide pitch, roll or yaw and watch the real LiDAR returns drift away from image edges. Restore the KITTI vendor calibration to compare.</p><p class="cloud-description">Manual projection experiment. Pixel displacement measures change from the vendor overlay; it is not calibration accuracy or a new check verdict. Occlusion and sensor timing can leave mismatches even at the vendor setting.</p><p class="cloud-source">KITTI raw · drive 0005<br>Geiger et al. · CC BY-NC-SA 3.0</p><a class="method-link" href="../assets/camera-lidar-live/">Source images, calibration &amp; processing ↗</a><a class="method-link" href="../assets/calibrex_check_camera_lidar/provenance.json">Earlier recorded rotation-check results ↗</a></aside>`;
      this.canvas=panel.querySelector("canvas");this.context=this.canvas.getContext("2d");
      AXES.forEach((axis,i)=>{
        const label=document.createElement("label");label.className="projection-axis";
        label.innerHTML=`<span>${axis} <output></output></span><input type="range" aria-label="Camera-LiDAR ${axis.toLowerCase()} offset" min="${i<3?-5:-.2}" max="${i<3?5:.2}" step="${i<3?.1:.01}" value="${this.offsets[i]}">`;
        label.querySelector("input").addEventListener("input",e=>{this.offsets[i]=Number(e.target.value);this.draw();});
        panel.querySelector(i<3?".projection-sliders":".projection-translation-sliders").append(label);
      });
      panel.querySelector("select").addEventListener("change",()=>this.load());
      panel.querySelector(".projection-retry").addEventListener("click",()=>this.load());
      panel.querySelector("#projection-split").addEventListener("input",e=>{this.split=Number(e.target.value);this.draw();});
      panel.querySelectorAll("[data-preset]").forEach(button=>button.addEventListener("click",()=>{
        this.offsets=[0,button.dataset.preset==="perturbed"?3:0,0,0,0,0];
        this.inputs().forEach((input,i)=>input.value=String(this.offsets[i]));this.draw();
      }));
    }
    inputs(){return [...this.panel.querySelectorAll(".projection-axis input")];}
    async load(){
      const seq=++this.sequence,panel=this.panel,frame=Number(panel.querySelector("select").value),status=panel.querySelector(".method-status");
      this.data=null;this.context.clearRect(0,0,this.canvas.width,this.canvas.height);
      panel.querySelector(".projection-content").hidden=true;panel.querySelector(".projection-retry").hidden=true;
      status.textContent="Loading real image and points…";
      try{
        if(![60,90].includes(frame))throw Error("Unknown camera-LiDAR scene");
        const data=validate(await this.json(`samples/camera_lidar_${String(frame).padStart(4,"0")}.json`));
        if(data.frame!==frame)throw Error("Mismatched camera-LiDAR scene");
        const image=new Image();image.src=`../assets/camera-lidar-live/frame_${String(frame).padStart(4,"0")}.jpg`;await image.decode();
        if(image.naturalWidth!==data.image_size[0]||image.naturalHeight!==data.image_size[1])throw Error("Image dimensions do not match the projection data");
        if(seq!==this.sequence)return;
        this.data=data;this.image=image;this.canvas.width=data.image_size[0];this.canvas.height=data.image_size[1];
        this.baseline=data.points.map(p=>project(p,data.intrinsics,[0,0,0,0,0,0]));
        panel.querySelector(".projection-content").hidden=false;this.draw();
        status.textContent="Real KITTI data · projection updates locally · no optimizer running";
      }catch(error){if(seq===this.sequence){status.textContent=error.message;panel.querySelector(".projection-retry").hidden=false;}}
    }
    draw(){
      if(!this.data)return;
      const ctx=this.context,w=this.canvas.width,h=this.canvas.height,split=w*this.split/100;
      const adjusted=this.data.points.map(p=>project(p,this.data.intrinsics,this.offsets));
      ctx.drawImage(this.image,0,0);ctx.fillStyle="#07101a40";ctx.fillRect(0,0,w,h);
      const inside=p=>p&&p[0]>=0&&p[0]<w&&p[1]>=0&&p[1]<h;
      for(const [points,left,right] of [[this.baseline,0,split],[adjusted,split,w]]){
        ctx.save();ctx.beginPath();ctx.rect(left,0,right-left,h);ctx.clip();
        for(const p of points){if(!inside(p))continue;const t=Math.max(0,Math.min(1,(p[2]-5)/35));ctx.fillStyle=`hsl(${25+165*t} 95% 62%)`;ctx.fillRect(p[0]-1,p[1]-1,2,2);}
        ctx.restore();
      }
      ctx.strokeStyle="#f8d288";ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(split,0);ctx.lineTo(split,h);ctx.stroke();
      const distances=[];
      adjusted.forEach((p,i)=>{const b=this.baseline[i];if(inside(p)&&inside(b))distances.push(Math.hypot(p[0]-b[0],p[1]-b[1]));});
      distances.sort((a,b)=>a-b);const mid=Math.floor(distances.length/2),median=distances.length?(distances.length%2?distances[mid]:(distances[mid-1]+distances[mid])/2):null;
      this.panel.querySelectorAll(".projection-axis output").forEach((o,i)=>o.textContent=`${this.offsets[i].toFixed(i<3?1:2)} ${i<3?"°":"m"}`);
      this.panel.querySelector(".projection-compare output").textContent=`${this.split}% vendor`;
      this.panel.querySelector(".projection-readouts").textContent=`${adjusted.filter(inside).length} visible points · ${distances.length} shared visible points · median displacement ${median===null?"unavailable":median.toFixed(1)+" px"}`;
      this.canvas.setAttribute("aria-label",`KITTI frame ${this.data.frame}: adjusted roll ${this.offsets[0]}, pitch ${this.offsets[1]}, yaw ${this.offsets[2]} degrees; translation ${this.offsets.slice(3).join(", ")} metres. Vendor overlay left of the divider.`);
    }
  }
  globalThis.CalibrexCameraLidar={validate,project,CameraLidarDemo};
})();
