"use strict";
(() => {
  const vector=v=>Array.isArray(v)&&v.length===3&&v.every(Number.isFinite);
  function validate(data){
    if(!data||data.schema!=="calibrex.camera_imu_timing_demo/v1"||data.capture_kind!=="real"||!Number.isFinite(data.saved_offset_s)||!Array.isArray(data.shift_range_ms)||data.shift_range_ms[0]!==-50||data.shift_range_ms[1]!==50)throw Error("Invalid measured clock data");
    const t=data.imu_time_s,c=data.gyro_cumulative_camera_deg,a=data.camera_start_s,b=data.camera_end_s,r=data.camera_rate_dps;
    if(!Array.isArray(t)||t.length<2||t.length>20000||!t.every((v,i)=>Number.isFinite(v)&&(i===0||v>t[i-1]))||!Array.isArray(c)||c.length!==t.length||!c.every(vector)||!Array.isArray(a)||!a.length||a.length>1000||!Array.isArray(b)||b.length!==a.length||!Array.isArray(r)||r.length!==a.length||!r.every(vector))throw Error("Invalid measured clock series");
    if(!a.every((v,i)=>Number.isFinite(v)&&Number.isFinite(b[i])&&b[i]>v&&(i===0||v>a[i-1])&&v+data.saved_offset_s-.05>=t[0]&&b[i]+data.saved_offset_s+.05<=t.at(-1)))throw Error("Clock range requires extrapolation");
    return data;
  }
  function cumulativeAt(t,times,values){
    if(t<times[0]||t>times.at(-1))throw Error("No IMU support at this time");
    let lo=0,hi=times.length-1;
    while(hi-lo>1){const m=(lo+hi)>>1;if(times[m]<=t)lo=m;else hi=m;}
    const f=(t-times[lo])/(times[hi]-times[lo]);
    return values[lo].map((v,k)=>v+f*(values[hi][k]-v));
  }
  function evaluate(data,shiftMs){
    if(!Number.isFinite(shiftMs)||shiftMs<-50||shiftMs>50)throw Error("Clock shift must be within ±50 ms");
    const offset=data.saved_offset_s+shiftMs/1000,predicted=[];
    data.camera_start_s.forEach((start,i)=>{
      const end=data.camera_end_s[i],a=cumulativeAt(start+offset,data.imu_time_s,data.gyro_cumulative_camera_deg),b=cumulativeAt(end+offset,data.imu_time_s,data.gyro_cumulative_camera_deg);
      predicted.push(a.map((v,k)=>(b[k]-v)/(end-start)));
    });
    return score(data,predicted,offset);
  }
  function score(data,predicted,offset){
    const axisSquares=[0,0,0],residuals=predicted.map((p,i)=>p.map((v,k)=>v-data.camera_rate_dps[i][k]));
    residuals.forEach(p=>p.forEach((v,k)=>axisSquares[k]+=v*v));
    return {offset_s:offset,predicted,residuals,rmse_dps:Math.sqrt(axisSquares.reduce((s,v)=>s+v,0)/predicted.length),axis_rmse_dps:axisSquares.map(v=>Math.sqrt(v/predicted.length))};
  }
  function rotateEvaluation(data,base,degrees,axis){
    if(!Number.isFinite(degrees)||degrees<-5||degrees>5||!Number.isInteger(axis)||axis<0||axis>2)throw Error("Rotation must be within ±5 degrees about X/Y/Z");
    const a=(axis+1)%3,b=(axis+2)%3,c=Math.cos(degrees*Math.PI/180),s=Math.sin(degrees*Math.PI/180);
    const predicted=base.predicted.map(p=>{const q=[...p];q[a]=c*p[a]-s*p[b];q[b]=s*p[a]+c*p[b];return q;});
    return score(data,predicted,base.offset_s);
  }
  function surface(data,profiles,axis){
    if(!Number.isInteger(axis)||axis<0||axis>2||profiles.length!==101)throw Error("Invalid rotation/time slice");
    const a=(axis+1)%3,b=(axis+2)%3,n=data.camera_rate_dps.length;
    // Aggregate the dot products once per time shift. The full grid then uses
    // norm invariance under rotation, without reintegrating or subsampling IMU.
    const moments=profiles.map(profile=>{
      let energy=0,parallel=0,cosine=0,sine=0;
      profile.predicted.forEach((g,i)=>{const c=data.camera_rate_dps[i];for(let k=0;k<3;k++)energy+=g[k]*g[k]+c[k]*c[k];parallel+=g[axis]*c[axis];cosine+=g[a]*c[a]+g[b]*c[b];sine+=g[a]*c[b]-g[b]*c[a];});
      return {energy:energy/n,parallel:parallel/n,cosine:cosine/n,sine:sine/n};
    });
    const angles=Array.from({length:41},(_,i)=>-5+i*.25),shifts=Array.from({length:101},(_,i)=>i-50);
    let minimum={rmse_dps:Infinity,shift_ms:0,rotation_deg:0};
    const rows=angles.map(degrees=>{const c=Math.cos(degrees*Math.PI/180),s=Math.sin(degrees*Math.PI/180);return moments.map((m,j)=>{
      const value=Math.sqrt(Math.max(0,m.energy-2*(m.parallel+c*m.cosine+s*m.sine)));
      if(value<minimum.rmse_dps)minimum={rmse_dps:value,shift_ms:shifts[j],rotation_deg:degrees};return value;
    });});
    return {angles,shifts,rows,minimum,similar_count:rows.flat().filter(v=>v<=minimum.rmse_dps+.5).length};
  }
  function surfacePoint(x,y){
    if(!Number.isFinite(x)||!Number.isFinite(y)||x<70||x>940||y<26||y>356)return null;
    const col=Math.max(0,Math.min(100,Math.round((x-70)/870*101-.5))),row=Math.max(0,Math.min(40,Math.round((356-y)/330*41-.5)));
    return {shift_ms:col-50,rotation_deg:-5+row*.25};
  }
  const COLOURS=["#5ce4d0","#ffba74"];
  class CameraImuTiming {
    constructor(panel,json){
      this.panel=panel;this.json=json;this.sequence=0;this.shift=0;this.axis=0;this.point=0;this.rotation=0;this.rotationAxis=0;
      panel.innerHTML=`<div class="method-media"><div class="cloud-topline"><span>REAL HILTI / CLOCK OFFSET EXPERIMENT</span><span>Saved baseline · fixed gyro bias</span></div><div class="timing-controls" hidden><div class="timing-toolbar"><label>Plot axis <select aria-label="Clock comparison axis"><option value="0">Camera X</option><option value="1">Camera Y</option><option value="2">Camera Z</option></select></label><strong class="timing-offset"></strong></div><label class="timing-shift-label">Injected shift relative to saved estimate <input type="range" min="-50" max="50" step="1" value="0" aria-label="Injected Camera-IMU clock shift"></label><div class="cloud-controls"><button type="button" data-shift="-50">−50 ms</button><button type="button" data-shift="-10">−10 ms</button><button type="button" data-shift="0">Reset both</button><button type="button" data-shift="10">+10 ms</button><button type="button" data-shift="50">+50 ms</button></div><div class="timing-rotation-controls"><label>Rotation axis <select aria-label="Injected rotation axis"><option value="0">Camera X</option><option value="1">Camera Y</option><option value="2">Camera Z</option></select></label><label>Injected camera-frame rotation <input type="range" min="-5" max="5" step=".25" value="0" aria-label="Injected Camera-IMU rotation"></label><output></output><button type="button" data-rotation-reset>Zero rotation</button></div><p class="timing-caption">3D residual RMSE · rotation × time slice · click a cell to inspect</p><canvas class="timing-surface" width="960" height="400" role="img" aria-label="Rotation and time offset residual heatmap"></canvas><div class="timing-surface-legend"><span class="timing-surface-low"></span><span class="timing-surface-colours"></span><span class="timing-surface-high"></span></div><p class="timing-caption">Cyan boundary: within +0.5 deg/s of this slice’s grid minimum. This is a residual contour, not a confidence region. Colour scale is fixed across rotation axes. White cross: selected. Cyan square: saved estimate.</p><div class="cloud-controls"><button type="button" data-slice-minimum>Inspect slice minimum</button><span class="timing-surface-detail"></span></div><p class="timing-caption">Camera and gyro curves are means over the same camera intervals · deg/s</p><canvas class="timing-rates" width="960" height="280" role="img" aria-label="Measured camera and interval-averaged gyro angular rates"></canvas><div class="timing-legend"><span>Camera-derived rate</span><span>Gyro mean in camera frame</span></div><p class="timing-caption">Selected-axis residual · gyro minus camera · deg/s</p><canvas class="timing-residuals" width="960" height="160" role="img" aria-label="Selected-axis angular-rate residual"></canvas><p class="timing-caption">3D residual RMSE versus time shift at the selected rotation · click to inspect</p><canvas class="timing-profile" width="960" height="170" role="img" aria-label="Angular-rate residual sensitivity to clock shift"></canvas><div class="timing-inspect"><label>Inspect interval <input type="range" min="0" max="0" value="0" aria-label="Clock comparison camera interval"></label><output></output></div></div><p class="method-status" role="status">Loading measured clock series…</p><button class="timing-retry" type="button" hidden>Retry clock data</button></div><aside class="cloud-evidence"><div class="eyebrow">CAMERA ↔ IMU / ROTATION AND CLOCK</div><h2>One residual.<br>Two parameters.</h2><p class="cloud-description">Vary time and one camera-frame rotation about the saved estimate. All other rotation components and the IMU-frame gyro bias remain fixed.</p><div class="timing-metrics"></div><p class="cloud-description timing-convention"></p><p class="cloud-description">Zero time and rotation injection restore the saved estimate, not ground truth. This development-window residual combines camera tracking noise and model errors. No optimizer or new calibration verdict is running.</p><p class="cloud-source">Hilti 2022 · exp21 / cam0<br>Hilti SLAM Challenge 2022 terms</p><a class="method-link" href="../assets/camera_imu_lock/timing/">Data, clock convention &amp; limits ↗</a><a class="method-link" href="../assets/camera-imu-lock.gif">Earlier recorded alignment replay ↗</a></aside>`;
      panel.querySelector("select").addEventListener("change",e=>{this.axis=Number(e.target.value);this.draw();});
      panel.querySelector(".timing-shift-label input").addEventListener("input",e=>this.setShift(Number(e.target.value)));
      panel.querySelectorAll("[data-shift]").forEach(b=>b.addEventListener("click",()=>Number(b.dataset.shift)===0?this.setParameters(0,0):this.setShift(Number(b.dataset.shift))));
      panel.querySelector('[aria-label="Injected rotation axis"]').addEventListener("change",e=>{this.rotationAxis=Number(e.target.value);this.draw();});
      panel.querySelector('[aria-label="Injected Camera-IMU rotation"]').addEventListener("input",e=>this.setParameters(this.shift,Number(e.target.value)));
      panel.querySelector("[data-rotation-reset]").addEventListener("click",()=>this.setParameters(this.shift,0));
      panel.querySelector("[data-slice-minimum]").addEventListener("click",()=>{if(this.data){const m=this.surfaces[this.rotationAxis].minimum;this.setParameters(m.shift_ms,m.rotation_deg);}});
      panel.querySelector(".timing-surface").addEventListener("click",e=>{if(!this.data)return;const box=e.currentTarget.getBoundingClientRect(),p=surfacePoint((e.clientX-box.left)/box.width*960,(e.clientY-box.top)/box.height*400);if(p)this.setParameters(p.shift_ms,p.rotation_deg);});
      panel.querySelector(".timing-inspect input").addEventListener("input",e=>{this.point=Number(e.target.value);this.draw();});
      panel.querySelector(".timing-retry").addEventListener("click",()=>this.load());
      panel.querySelector(".timing-profile").addEventListener("click",e=>{if(!this.data)return;const box=e.currentTarget.getBoundingClientRect(),x=(e.clientX-box.left)/box.width*960;this.setShift(Math.max(-50,Math.min(50,Math.round(-50+(x-58)/872*100))));});
    }
    async load(){
      const seq=++this.sequence,panel=this.panel;this.data=null;panel.querySelector(".timing-controls").hidden=true;panel.querySelector(".timing-retry").hidden=true;
      panel.querySelector(".method-status").textContent="Loading measured clock series…";
      panel.querySelector(".timing-metrics").replaceChildren();
      try{
        const data=validate(await this.json("samples/camera_imu_timing.json?v=rotation-surface-1"));if(seq!==this.sequence)return;
        this.data=data;this.profile=Array.from({length:101},(_,i)=>evaluate(data,i-50));this.rotationProfileKey=null;
        this.surfaces=[0,1,2].map(k=>surface(data,this.profile,k));
        const cells=this.surfaces.flatMap(s=>s.rows.flat());this.colourBounds=[Math.min(...cells),Math.max(...cells)];
        this.point=Math.min(this.point,data.camera_start_s.length-1);panel.querySelector(".timing-inspect input").max=String(data.camera_start_s.length-1);
        const angularBound=2*Math.sin(5*Math.PI/360);
        this.rateBounds=[0,1,2].map(k=>{const values=[...data.camera_rate_dps.map(p=>p[k]),...this.profile.flatMap(r=>r.predicted.flatMap(p=>{const delta=angularBound*Math.hypot(...p);return [p[k]-delta,p[k]+delta];}))];return [Math.min(...values),Math.max(...values)];});
        this.residualBounds=[0,1,2].map(k=>{const bound=Math.max(...this.profile.flatMap(r=>r.predicted.map((p,i)=>Math.abs(r.residuals[i][k])+angularBound*Math.hypot(...p))));return [-bound,bound];});
        panel.querySelector(".timing-controls").hidden=false;this.setShift(this.shift);
        panel.querySelector(".timing-convention").textContent=`t_IMU = t_camera + ${(data.saved_offset_s*1000).toFixed(3)} ms + injected shift. A positive shift samples later IMU times. All ${data.camera_start_s.length} intervals remain the same; no extrapolation.`;
        panel.querySelector(".method-status").textContent="Measured Hilti series loaded · interval means recomputed locally · no optimizer running";
      }catch(error){if(seq===this.sequence){panel.querySelector(".method-status").textContent=error.message;panel.querySelector(".timing-retry").hidden=false;}}
    }
    setShift(value){this.setParameters(value,this.rotation);}
    setParameters(shift,rotation){this.shift=shift;this.rotation=rotation;this.panel.querySelector(".timing-shift-label input").value=String(shift);this.panel.querySelector('[aria-label="Injected Camera-IMU rotation"]').value=String(rotation);this.draw();}
    rotatedProfile(){
      const key=`${this.rotationAxis}:${this.rotation}`;
      if(this.rotationProfileKey!==key){this.rotationProfile=this.profile.map(base=>rotateEvaluation(this.data,base,this.rotation,this.rotationAxis));this.rotationProfileKey=key;}
      return this.rotationProfile;
    }
    draw(){
      if(!this.data)return;
      const d=this.data,k=this.axis,profile=this.rotatedProfile(),result=profile[this.shift+50],baseline=this.profile[50],axis=["X","Y","Z"][k],n=d.camera_start_s.length;
      this.drawSurface();
      this.panel.querySelector(".timing-rotation-controls output").textContent=`${this.rotation>=0?"+":""}${this.rotation.toFixed(2)}° about camera ${["X","Y","Z"][this.rotationAxis]}`;
      this.panel.querySelector(".timing-offset").textContent=`${this.shift>=0?"+":""}${this.shift} ms injected`;
      const x=d.camera_start_s.map((v,i)=>(v+d.camera_end_s[i])/2),camera=d.camera_rate_dps.map(v=>v[k]),gyro=result.predicted.map(v=>v[k]),residual=result.residuals.map(v=>v[k]);
      this.plot(".timing-rates",x,[camera,gyro],this.rateBounds[k],this.point,"s","deg/s");
      this.plot(".timing-residuals",x,[residual],this.residualBounds[k],this.point,"s","deg/s");
      const profileValues=profile.map(v=>v.rmse_dps);
      this.plot(".timing-profile",this.profile.map((_,i)=>i-50),[profileValues],this.colourBounds,this.shift+50,"ms","deg/s");
      const rows=[["Displayed-window 3D RMSE",result.rmse_dps.toFixed(3)+" deg/s"],["Saved estimate · zero injections",baseline.rmse_dps.toFixed(3)+" deg/s"],[`Selected ${axis}-axis RMSE`,result.axis_rmse_dps[k].toFixed(3)+" deg/s"],["Total candidate time offset",(result.offset_s*1000).toFixed(3)+" ms"],["Same camera intervals",String(n)],["Near-minimum cells · +0.5 deg/s",`${this.surfaces[this.rotationAxis].similar_count} / 4141`]];
      const metrics=this.panel.querySelector(".timing-metrics");metrics.replaceChildren();
      for(const [label,value] of rows){const card=document.createElement("div"),small=document.createElement("small"),strong=document.createElement("strong");small.textContent=label;strong.textContent=value;card.append(small,strong);metrics.append(card);}
      this.panel.querySelector(".timing-inspect input").value=String(this.point);
      this.panel.querySelector(".timing-inspect output").textContent=`Interval ${this.point+1}/${n}: ${d.camera_start_s[this.point].toFixed(3)}–${d.camera_end_s[this.point].toFixed(3)} s · ${axis}: camera ${camera[this.point].toFixed(2)}, gyro ${gyro[this.point].toFixed(2)}, residual ${residual[this.point].toFixed(2)} deg/s`;
      this.panel.querySelector(".timing-rates").setAttribute("aria-label",`Camera ${axis} interval-average rates, injected offset ${this.shift} milliseconds and rotation ${this.rotation} degrees about camera ${["X","Y","Z"][this.rotationAxis]}, displayed-window 3D RMSE ${result.rmse_dps.toFixed(3)} degrees per second.`);
    }
    drawSurface(){
      const map=this.surfaces[this.rotationAxis],canvas=this.panel.querySelector(".timing-surface"),ctx=canvas.getContext("2d"),[lo,hi]=this.colourBounds;
      ctx.fillStyle="#091524";ctx.fillRect(0,0,960,400);
      const px=j=>70+j/101*870,py=i=>26+(40-i)/41*330;
      const similar=(i,j)=>i>=0&&i<41&&j>=0&&j<101&&map.rows[i][j]<=map.minimum.rmse_dps+.5;
      for(let i=0;i<41;i++)for(let j=0;j<101;j++){
        const t=Math.max(0,Math.min(1,(map.rows[i][j]-lo)/(hi-lo||1)));
        ctx.fillStyle=`rgb(${Math.round(20+220*t)} ${Math.round(45+95*t)} ${Math.round(75-35*t)})`;ctx.fillRect(px(j),py(i),870/101+1,330/41+1);
      }
      ctx.strokeStyle="#5ce4d0";ctx.lineWidth=1;
      for(let i=0;i<41;i++)for(let j=0;j<101;j++)if(similar(i,j)){
        const x=px(j),y=py(i),w=870/101,h=330/41;ctx.beginPath();
        if(!similar(i+1,j)){ctx.moveTo(x,y);ctx.lineTo(x+w,y);}if(!similar(i-1,j)){ctx.moveTo(x,y+h);ctx.lineTo(x+w,y+h);}
        if(!similar(i,j-1)){ctx.moveTo(x,y);ctx.lineTo(x,y+h);}if(!similar(i,j+1)){ctx.moveTo(x+w,y);ctx.lineTo(x+w,y+h);}ctx.stroke();
      }
      const x=px(this.shift+50+.5),y=py(Math.round((this.rotation+5)/.25))+330/82;
      ctx.strokeStyle="#fff";ctx.lineWidth=2;ctx.beginPath();ctx.arc(x,y,6,0,2*Math.PI);ctx.moveTo(x-11,y);ctx.lineTo(x+11,y);ctx.moveTo(x,y-11);ctx.lineTo(x,y+11);ctx.stroke();
      ctx.strokeStyle="#5ce4d0";ctx.lineWidth=1.5;const bx=px(50.5),by=py(20)+330/82;ctx.strokeRect(bx-4,by-4,8,8);
      ctx.fillStyle="#b9cede";ctx.font="13px monospace";ctx.fillText("+5°",18,36);ctx.fillText("0°",24,197);ctx.fillText("−5°",18,357);ctx.fillText("−50 ms",70,385);ctx.fillText("0 ms",480,385);ctx.fillText("+50 ms",875,385);ctx.fillText(`Camera ${["X","Y","Z"][this.rotationAxis]} rotation`,72,17);
      this.panel.querySelector(".timing-surface-low").textContent=lo.toFixed(2)+" deg/s";this.panel.querySelector(".timing-surface-high").textContent=hi.toFixed(2)+" deg/s";
      const m=map.minimum;this.panel.querySelector(".timing-surface-detail").textContent=`Grid minimum: ${m.rmse_dps.toFixed(3)} deg/s at ${m.rotation_deg.toFixed(2)}° / ${m.shift_ms} ms · diagnostic only`;
      canvas.setAttribute("aria-label",`Camera ${["X","Y","Z"][this.rotationAxis]} rotation versus time offset. Minimum displayed grid residual ${m.rmse_dps.toFixed(3)} degrees per second. Selected ${this.rotation} degrees and ${this.shift} milliseconds. Equivalent keyboard controls are the rotation and clock sliders.`);
    }
    plot(selector,x,series,bounds,cursor,xunit,yunit){
      const canvas=this.panel.querySelector(selector),ctx=canvas.getContext("2d"),w=canvas.width,h=canvas.height;
      const [lo,hi]=bounds,span=hi-lo||1,xmin=x[0],xmax=x.at(-1),px=v=>58+(v-xmin)/(xmax-xmin||1)*(w-88),py=v=>h-34-(v-lo+span*.1)/(span*1.2)*(h-62);
      ctx.fillStyle="#091524";ctx.fillRect(0,0,w,h);ctx.strokeStyle="#24394e";ctx.lineWidth=1;
      for(let i=0;i<=4;i++){const yy=28+i*(h-62)/4;ctx.beginPath();ctx.moveTo(58,yy);ctx.lineTo(w-30,yy);ctx.stroke();}
      if(lo<=0&&hi>=0){ctx.strokeStyle="#8eabc177";ctx.beginPath();ctx.moveTo(58,py(0));ctx.lineTo(w-30,py(0));ctx.stroke();}
      series.forEach((values,k)=>{ctx.strokeStyle=COLOURS[k];ctx.lineWidth=2;ctx.beginPath();values.forEach((v,i)=>i?ctx.lineTo(px(x[i]),py(v)):ctx.moveTo(px(x[i]),py(v)));ctx.stroke();});
      ctx.strokeStyle="#fff8";ctx.beginPath();ctx.moveTo(px(x[cursor]),18);ctx.lineTo(px(x[cursor]),h-30);ctx.stroke();
      ctx.fillStyle="#a7bfd4";ctx.font="12px monospace";ctx.fillText(hi.toFixed(1),5,27);ctx.fillText(lo.toFixed(1),5,h-29);ctx.fillText(`${xmin.toFixed(1)} ${xunit}`,58,h-8);ctx.fillText(`${xmax.toFixed(1)} ${xunit}`,w-110,h-8);ctx.fillText(yunit,w-80,18);
    }
  }
  globalThis.CalibrexTiming={validate,cumulativeAt,evaluate,rotateEvaluation,surface,surfacePoint,CameraImuTiming};
})();
