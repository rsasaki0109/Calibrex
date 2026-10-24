"use strict";
(() => {
  const METHODS = [
    {id:"lidar",label:"LiDAR ↔ LiDAR",tag:"Point clouds"},
    {id:"deskew",label:"IMU ↔ LiDAR",tag:"Motion deskew",title:"Straighten a moving sweep.",
      description:"One MID360 sweep during a 180°/s turn. The IMU-LiDAR estimate compensates rotation using each point’s measurement time; translation is not compensated.",
      image:"../assets/mid360-deskew.gif",source:"../assets/mid360-deskew/manifest.json",dataset:"RTK-SLAM · construction_seq1",license:"CC BY 4.0",
      metric:"Local surface thickness · proxy, not accuracy",unit:"cm",before:"Raw sweep",after:"Gyro deskewed"},
    {id:"camera-imu",label:"Camera ↔ IMU",tag:"Interactive clock offset",title:"Make the rate curves agree.",
      description:"Camera-derived angular rates and gyro rates in the camera frame, on Hilti development data. Rotation, clock offset and bias move between a perturbed hypothesis and the saved estimate.",
      image:"../assets/camera-imu-lock.gif",source:"../assets/camera_imu_lock/series.provenance.json",dataset:"Hilti 2022 · exp21 / cam0",license:"Hilti SLAM Challenge 2022 terms",
      metric:"Displayed-interval angular-rate RMSE",unit:"deg/s",before:"Perturbed hypothesis",after:"Saved estimate"},
    {id:"camera-lidar",label:"Camera ↔ LiDAR",tag:"Interactive projection",title:"Put the points on the objects.",
      description:"Two real KITTI camera-LiDAR checks: a +3° injected camera rotation and the vendor transform. Only roll and pitch are judged here; translation remains unchecked.",
      image:"../assets/calibrex-check-camera-lidar.gif",source:"../assets/calibrex_check_camera_lidar/provenance.json",dataset:"KITTI · drive 0005",license:"CC BY-NC-SA 3.0",
      metric:"Pitch error against the estimate",unit:"deg",before:"Injected rotation",after:"Vendor transform"},
    {id:"stereo",label:"Fisheye stereo",tag:"Real stereo photos"},
    {id:"comparison",label:"Method comparison",tag:"4 board / 2 targetless"},
    {id:"handeye",label:"Robot hand–eye",tag:"8 real-data methods"},
    {id:"gnss",label:"GNSS tracks",tag:"Measured receiver fixes"},
    {id:"vehicle",label:"Wheels / INS",tag:"4 measured wheel speeds"},
    {id:"radar",label:"Radar tracks",tag:"Measured range & speed"},
    {id:"live",label:"Live calibration",tag:"Actual optimizer updates"},
  ];
  const validPNG = (value) => typeof value==="string" && value.length<1800000 && /^data:image\/png;base64,[A-Za-z0-9+/=]+$/.test(value);
  function validateStereo(data){
    if(!data||!Number.isInteger(data.matches)||data.matches<0||![data.vertical_median_px,data.baseline_m,data.focal_px].every(v=>Number.isFinite(v)&&v>=0)||data.baseline_m<=0||data.focal_px<=0||![data.raw,data.rectified,data.depth].every(validPNG))throw Error("Invalid stereo demo data");
    if(data.capture_kind!=="real"&&(!Number.isFinite(data.median_relative_depth_error)||data.median_relative_depth_error<0))throw Error("Invalid stereo demo data");
    const size=data.image_size,points=data.correspondences;
    if(!Array.isArray(size)||size.length!==2||!size.every(v=>Number.isInteger(v)&&v>0&&v<=4096)||!Array.isArray(points)||points.length!==data.matches||points.length===0||points.length>5000)throw Error("Invalid stereo correspondences");
    const inside=p=>Array.isArray(p)&&p.length===2&&p.every((v,i)=>Number.isFinite(v)&&v>=0&&v<size[i]);
    for(const p of points){
      if(!p||!inside(p.left)||!inside(p.right)||!Number.isFinite(p.disparity_px)||p.disparity_px<=0||!Number.isFinite(p.depth_m)||p.depth_m<=0||Math.abs(p.left[0]-p.right[0]-p.disparity_px)>1e-6||Math.abs(data.focal_px*data.baseline_m/p.disparity_px-p.depth_m)>1e-6)throw Error("Invalid stereo correspondences");
    }
    return data;
  }
  // Pixel coordinates refer to pixel centres; clip each view separately so a
  // crop at the camera boundary never samples the other half of the pair.
  function stereoCrop(point,width,height){
    if(!Array.isArray(point)||point.length!==2||!point.every(Number.isFinite)||!Number.isInteger(width)||!Number.isInteger(height)||width<=0||height<=0||point[0]<0||point[0]>=width||point[1]<0||point[1]>=height)throw Error("Invalid stereo crop");
    const [x,y]=point,x0=Math.floor(x)-20,y0=Math.floor(y)-20,sx=Math.max(0,x0),sy=Math.max(0,y0),sw=Math.min(width,x0+41)-sx,sh=Math.min(height,y0+41)-sy;
    return {sx,sy,sw,sh,dx:sx-x0,dy:sy-y0,cx:x-x0+.5,cy:y-y0+.5,size:41};
  }
  class MethodGallery {
    constructor(root,cloud){
      this.root=root;this.cloud=cloud;this.current="lidar";this.request=0;this.frameRequest=0;this.cache=new Map();this.evidence=new Map();
      this.panels=root.querySelector(".method-panels");this.tabs=root.querySelector(".method-tabs");
      for(const method of METHODS){
        const button=document.createElement("button");button.type="button";button.role="tab";button.id="demo-tab-"+method.id;button.dataset.method=method.id;
        button.innerHTML=`<b>${method.label}</b><small>${method.tag}</small>`;
        button.addEventListener("click",()=>this.select(method.id));this.tabs.append(button);
        if(method.id!=="lidar")this.panels.append(this.panel(method));
      }
      this.tabs.addEventListener("keydown",(e)=>{
        if(!["ArrowLeft","ArrowRight","Home","End"].includes(e.key))return;e.preventDefault();
        let i=METHODS.findIndex((m)=>m.id===this.current);i=e.key==="Home"?0:e.key==="End"?METHODS.length-1:(i+(e.key==="ArrowRight"?1:-1)+METHODS.length)%METHODS.length;
        this.select(METHODS[i].id);this.tabs.children[i].focus();
      });
      this.select("lidar");
    }
    panel(method){
      const panel=document.createElement("section");panel.id="demo-panel-"+method.id;panel.className="alignment-demo method-panel";panel.hidden=true;panel.role="tabpanel";
      panel.setAttribute("aria-labelledby","demo-tab-"+method.id);
      if(CalibrexEvidence.CODES.includes(method.id)){
        this.evidence.set(method.id,new CalibrexEvidence.EvidenceDemo(panel,url=>this.json(url),method.id));
      }else if(method.id==="camera-imu"){
        this.cameraImu=new CalibrexTiming.CameraImuTiming(panel,url=>this.json(url));
      }else if(method.id==="camera-lidar"){
        this.cameraLidar=new CalibrexCameraLidar.CameraLidarDemo(panel,url=>this.json(url));
      }else if(method.id==="stereo"){
        panel.innerHTML=`<div class="method-media"><div class="cloud-topline"><span>REAL CAPTURE / NATIVE RECTIFICATION</span><span>KB4 · calibrated stereo rig</span></div><div class="stereo-picker"><label>Camera model <select class="stereo-model-select" aria-label="Real fisheye camera model"><option value="kb4">KB4 · upstream calibration</option><option value="mei">MEI · fit to measured corners</option><option value="ds">Double Sphere · fit to measured corners</option></select></label><label>Photographed pair <select class="stereo-frame-select" aria-label="Real fisheye stereo pair"><option value="000">Pair 000 · board in the room</option><option value="014">Pair 014 · board close up</option><option value="027">Pair 027 · board tilted</option></select></label></div><div class="stereo-views"><figure><figcaption>RAW · LEFT / RIGHT</figcaption><img class="stereo-raw" alt="Real photographed left and right fisheye images"></figure><figure><figcaption>RECTIFIED · EPIPOLAR GUIDES</figcaption><div class="stereo-overlay"><img class="stereo-rectified" alt="Native rectified images with horizontal guides"><svg class="stereo-pair-marker" aria-hidden="true"></svg></div></figure><figure class="stereo-depth-figure"><figcaption>SPARSE MATCHED DEPTH · NEAR <span class="depth-key"></span> FAR</figcaption><div class="stereo-depth-row"><div class="stereo-overlay stereo-depth-map"><img class="stereo-depth" alt="Depth from image patch matches; colored sparse samples"><svg class="stereo-depth-marker" aria-hidden="true"></svg></div><p>Click a depth dot to inspect its match in both rectified images.<br><br>Orange: ≤ 0.3 m<br>Cyan: ≥ 3.0 m<br>Colour clips at both ends<br><br>Use the point slider or arrow keys to explore every accepted match.</p></div></figure><div class="stereo-cloud-panel"></div><div class="stereo-inspector"><div class="stereo-point-controls"><label for="stereo-point">Inspect match <output class="stereo-point-number"></output></label><input id="stereo-point" type="range" min="0" max="0" value="0" aria-label="Stereo match"><button type="button" class="stereo-previous">← Previous match</button><button type="button" class="stereo-next">Next match →</button></div><div class="stereo-patch-header"><strong>Look closer at the match</strong><span>41 × 41 px · linked views</span></div><div class="stereo-patches" hidden><figure><figcaption><b>LEFT</b><span class="stereo-left-coordinate"></span></figcaption><canvas class="stereo-left-patch" width="246" height="246" role="img" aria-label="Left correspondence close-up"></canvas></figure><figure><figcaption><b>RIGHT</b><span class="stereo-right-coordinate"></span></figcaption><canvas class="stereo-right-patch" width="246" height="246" role="img" aria-label="Right correspondence close-up"></canvas></figure></div><p class="stereo-patch-note">Crosses mark the saved correspondence; boxes show an 11 × 11 px neighbourhood. Compare texture on both sides: repeated squares can still give a wrong match. Saved epipolar guides remain visible. No new match or confidence score is computed.</p><div class="stereo-measurements" aria-live="polite"></div><p class="stereo-equation"></p></div></div><p class="method-status" role="status">Loading photographed pair…</p></div><aside class="cloud-evidence"><div class="eyebrow">FISHEYE STEREO / REAL PHOTOGRAPHS</div><h2>Real lenses.<br>Real room.</h2><p class="cloud-description">Three photographed stereo pairs from OpenCV’s fisheye test data. KB4 uses the supplied calibration; MEI and Double Sphere use new fits to measured board corners. Native ZNCC patch matches estimate depth. All three models support all three photographed pairs.</p><div class="stereo-stats"></div><p class="cloud-description">Estimated depth, without independent depth truth. Repeated board squares and occlusions can produce wrong matches. Treat these saved estimates as a visual demonstration. Depth is Z = fB/d.</p><a class="method-link" href="../tutorials/fisheye_stereo/">Models, API &amp; limitations ↗</a><a class="method-link" href="../assets/fisheye-stereo-real/">Photographs, calibration &amp; attribution ↗</a></aside>`;
        this.stereoCloud=new CalibrexStereoCloud.StereoCloud(panel.querySelector(".stereo-cloud-panel"),id=>this.inspectPoint(id));
        panel.querySelector("#stereo-point").addEventListener("input",e=>this.inspectPoint(Number(e.target.value)));
        panel.querySelector(".stereo-rectified").addEventListener("load",()=>this.drawStereoPatches());
        panel.querySelector(".stereo-previous").addEventListener("click",()=>{if(this.stereoData)this.inspectPoint((this.pointIndex-1+this.stereoData.matches)%this.stereoData.matches);});
        panel.querySelector(".stereo-next").addEventListener("click",()=>{if(this.stereoData)this.inspectPoint((this.pointIndex+1)%this.stereoData.matches);});
        panel.querySelector(".stereo-depth-map").addEventListener("click",e=>{
          if(!this.stereoData)return;
          const box=e.currentTarget.getBoundingClientRect(),[w,h]=this.stereoData.image_size;
          const x=(e.clientX-box.left)/box.width*w,y=(e.clientY-box.top)/box.height*h;
          let nearest=0,distance=Infinity;
          this.stereoData.correspondences.forEach((p,i)=>{const d=(p.left[0]-x)**2+(p.left[1]-y)**2;if(d<distance){nearest=i;distance=d;}});
          this.inspectPoint(nearest);
        });
        panel.querySelector(".stereo-frame-select").addEventListener("change",e=>this.loadFrame(e.target.value));
        panel.querySelector(".stereo-model-select").addEventListener("change",()=>this.loadFrame(panel.querySelector(".stereo-frame-select").value));
      }else{
        panel.innerHTML=`<div class="method-media"><div class="cloud-topline"><span>REAL RECORDING / SAVED COMPARISON</span><span>${method.dataset}</span></div><div class="method-replay"><img hidden alt="${method.label} recorded evidence visualization"><div class="method-poster"><span>${method.tag}</span><strong>${method.title}</strong><p>Play the recorded data visualization.</p></div></div><div class="cloud-controls"><button type="button" data-replay>Play replay</button><span class="method-format">Recorded visualization · not live optimization</span></div><p class="method-status" role="status"></p></div><aside class="cloud-evidence"><div class="eyebrow">${method.label}</div><h2>${method.title}</h2><p class="cloud-description">${method.description}</p><div class="cloud-metric-label">${method.metric}</div><div class="cloud-metric"><span class="method-before">—</span><small>${method.unit}</small></div><span class="cloud-metric-caption">${method.before}</span><div class="cloud-metric cloud-metric-after"><span class="method-after">—</span><small>${method.unit}</small></div><span class="cloud-metric-caption">${method.after}</span><p class="cloud-source">${method.dataset}<br>${method.license}</p><p class="cloud-description">Saved endpoints. Intermediate frames illustrate parameter changes or overlays; they are not optimizer iterations or new check results.</p><a class="method-link" href="${method.source}">Source, metrics &amp; caveats ↗</a></aside>`;
        panel.querySelector("[data-replay]").addEventListener("click",()=>{if(panel.querySelector("img").hidden)this.play(method);else this.pause(panel);});
        panel.querySelector("img").addEventListener("error",()=>{this.pause(panel);panel.querySelector(".method-status").textContent="The recorded visualization could not load. Select Play replay to retry.";});
      }
      return panel;
    }
    async json(url){
      if(!this.cache.has(url)){
        const promise=fetch(url).then(r=>{if(!r.ok)throw Error(`Demo data unavailable (${r.status})`);return r.json();});
        this.cache.set(url,promise);promise.catch(()=>this.cache.delete(url));
      }
      return this.cache.get(url);
    }
    async select(id){
      const method=METHODS.find(m=>m.id===id);if(!method)return;
      this.request++;const request=this.request;this.current=id;this.cloud.stop();this.evidence.forEach(d=>d.stop());
      for(const panel of this.panels.children){panel.hidden=panel.id!=="demo-panel-"+id && !(id==="lidar"&&panel.id==="alignment-demo");if(panel.hidden&&panel.querySelector("[data-replay]"))this.pause(panel);}
      for(const button of this.tabs.children){const selected=button.dataset.method===id;button.setAttribute("aria-selected",String(selected));button.tabIndex=selected?0:-1;}
      const panel=id==="lidar"?this.root.querySelector("#alignment-demo"):this.root.querySelector("#demo-panel-"+id);
      panel.setAttribute("aria-labelledby","demo-tab-"+id);panel.role="tabpanel";
      if(id==="lidar")return;
      if(this.evidence.has(id)){this.evidence.get(id).load();return;}
      if(id==="camera-lidar"){this.cameraLidar.load();return;}
      if(id==="camera-imu"){this.cameraImu.load();return;}
      if(id==="stereo"){this.loadFrame(panel.querySelector(".stereo-frame-select").value);return;}
      try{
        let before,after;
        if(id==="deskew"){const d=await this.json(method.source);before=d.metrics.thickness_raw_cm;after=d.metrics.thickness_deskewed_cm;}
        else if(id==="camera-imu"){const d=await this.json("samples/method_demo_metrics.json");before=d.camera_imu.before;after=d.camera_imu.after;}
        else {const d=await this.json(method.source);before=d.check_runs.perturbed.axes.pitch.candidate_error_deg;after=d.check_runs.reference.axes.pitch.candidate_error_deg;}
        if(request!==this.request)return;
        if(![before,after].every(Number.isFinite))throw Error("Missing demo metrics");
        panel.querySelector(".method-before").textContent=Math.abs(before).toFixed(id==="deskew"?2:1);
        panel.querySelector(".method-after").textContent=Math.abs(after).toFixed(id==="deskew"?2:1);
        if(!matchMedia("(prefers-reduced-motion: reduce)").matches)this.play(method);
      }catch(error){if(request===this.request)panel.querySelector(".method-status").textContent=error.message;}
    }
    play(method){const panel=this.root.querySelector("#demo-panel-"+method.id),img=panel.querySelector("img");img.src=method.image;img.hidden=false;panel.querySelector(".method-poster").hidden=true;panel.querySelector("[data-replay]").textContent="Stop replay";}
    pause(panel){const img=panel.querySelector("img");img.hidden=true;img.removeAttribute("src");panel.querySelector(".method-poster").hidden=false;panel.querySelector("[data-replay]").textContent="Play replay";}
    async loadFrame(code){
      const panel=this.root.querySelector("#demo-panel-stereo"),seq=++this.frameRequest,status=panel.querySelector(".method-status");status.textContent="Loading photographed pair…";this.stereoData=null;this.stereoCloud.clear();panel.querySelector(".stereo-stats").textContent="";
      panel.querySelector(".stereo-views").hidden=true;panel.querySelector(".stereo-patches").hidden=true;
      try{
        const index=await this.json("samples/fisheye_real_demo.json?v=three-pairs-1"),model=panel.querySelector(".stereo-model-select").value;
        if(!["kb4","mei","ds"].includes(model)||!["000","014","027"].includes(code)||index.schema!=="calibrex.real_fisheye_stereo_demo/v1"||index.capture_kind!=="real"||!index.models[model]?.includes(code))throw Error("Unknown real stereo pair/model");
        const suffix=model==="kb4"?code:model+"_"+code;
        const data=validateStereo(await this.json("samples/fisheye_real_"+suffix+".json?v=stable-pixels-1"));if(seq!==this.frameRequest)return;
        if(data.camera_model!==model)throw Error("Mismatched stereo model");
        if(data.capture_kind!=="real"||data.frame!==code||data.depth_truth!==null)throw Error("Invalid real stereo receipt");
        for(const key of ["raw","rectified","depth"])panel.querySelector(".stereo-"+key).src=data[key];
        panel.querySelector(".stereo-stats").textContent=`${data.matches} image matches\nMedian vertical mismatch: ${data.vertical_median_px.toFixed(3)} px\nCalibrated baseline: ${(1000*data.baseline_m).toFixed(1)} mm\nDepth accuracy: not measured`;
        panel.querySelector(".cloud-topline span:last-child").textContent=`${model.toUpperCase()} · photographed pair ${code}`;
        this.stereoData=data;this.stereoCloud.setData(data);
        panel.querySelector("#stereo-point").max=String(data.matches-1);
        this.inspectPoint(0);
        panel.querySelector(".stereo-views").hidden=false;status.textContent=model==="kb4"?"Real OpenCV photographs · saved native rectification and sparse depth":`Real OpenCV pair ${code} · ${model.toUpperCase()} fitted to measured corners · fit-level excluded-capture right reprojection ${data.fit_excluded_reprojection_px.toFixed(3)} px · KB4 initialized the fit; this is not an independent benchmark`;
      }catch(error){if(seq===this.frameRequest)status.textContent=error.message;}
    }
    inspectPoint(index){
      const data=this.stereoData;if(!data||!Number.isInteger(index)||index<0||index>=data.matches)return;
      this.pointIndex=index;this.stereoCloud.select(index);
      const panel=this.root.querySelector("#demo-panel-stereo"),p=data.correspondences[index],[w,h]=data.image_size;
      const [lx,ly]=p.left,[rx,ry]=p.right;
      const pair=panel.querySelector(".stereo-pair-marker"),depth=panel.querySelector(".stereo-depth-marker");
      pair.setAttribute("viewBox",`0 0 ${2*w} ${h}`);depth.setAttribute("viewBox",`0 0 ${w} ${h}`);
      pair.innerHTML=`<path d="M 0 ${ly} H ${2*w}" class="match-guide"/><path d="M ${lx} ${ly} L ${w+rx} ${ry}" class="match-connection"/><circle cx="${lx}" cy="${ly}" r="7"/><circle cx="${w+rx}" cy="${ry}" r="7"/>`;
      depth.innerHTML=`<circle cx="${lx}" cy="${ly}" r="8"/>`;
      panel.querySelector("#stereo-point").value=String(index);
      panel.querySelector(".stereo-point-number").textContent=`${index+1} / ${data.matches}`;
      panel.querySelector(".stereo-measurements").innerHTML=`<div><small>DISPARITY</small><strong>${p.disparity_px.toFixed(0)}<span> px</span></strong></div><div><small>ESTIMATED Z</small><strong>${p.depth_m.toFixed(2)}<span> m</span></strong></div><div><small>VERTICAL MISMATCH</small><strong>${Math.abs(ry-ly).toFixed(3)}<span> px</span></strong></div>`;
      this.drawStereoPatches();
      panel.querySelector(".stereo-equation").textContent=`Z = f × B / d = ${data.focal_px.toFixed(1)} px × ${data.baseline_m.toFixed(4)} m / ${p.disparity_px.toFixed(0)} px = ${p.depth_m.toFixed(2)} m · rectified left-camera depth`;
    }
    drawStereoPatches(){
      const data=this.stereoData;if(!data)return;
      const panel=this.root.querySelector("#demo-panel-stereo"),image=panel.querySelector(".stereo-rectified"),[w,h]=data.image_size;
      if(!image.complete||image.naturalWidth!==2*w||image.naturalHeight!==h||image.getAttribute("src")!==data.rectified)return;
      const point=data.correspondences[this.pointIndex];if(!point)return;
      for(const [side,offset,colour] of [["left",0,"#ffcb81"],["right",w,"#65e1cb"]]){
        const p=point[side],g=stereoCrop(p,w,h),canvas=panel.querySelector(".stereo-"+side+"-patch"),ctx=canvas.getContext("2d");
        ctx.setTransform(1,0,0,1,0,0);ctx.fillStyle="#08121f";ctx.fillRect(0,0,246,246);ctx.scale(6,6);ctx.imageSmoothingEnabled=false;
        ctx.drawImage(image,offset+g.sx,g.sy,g.sw,g.sh,g.dx,g.dy,g.sw,g.sh);
        ctx.strokeStyle=colour;ctx.lineWidth=1/3;ctx.strokeRect(g.cx-5.5,g.cy-5.5,11,11);
        ctx.lineWidth=1/6;ctx.beginPath();ctx.moveTo(g.cx-3,g.cy);ctx.lineTo(g.cx+3,g.cy);ctx.moveTo(g.cx,g.cy-3);ctx.lineTo(g.cx,g.cy+3);ctx.stroke();
        panel.querySelector(".stereo-"+side+"-coordinate").textContent=`x ${p[0].toFixed(2)} · y ${p[1].toFixed(2)} px`;
        canvas.setAttribute("aria-label",`${side} rectified display close-up, match ${this.pointIndex+1} of ${data.matches}, x ${p[0].toFixed(2)}, y ${p[1].toFixed(2)} pixels. Estimated depth ${point.depth_m.toFixed(2)} metres, without independent depth truth.`);
      }
      panel.querySelector(".stereo-patches").hidden=false;
    }
    async explore(){
      const id=this.current,panel=id==="lidar"?this.root.querySelector("#alignment-demo"):this.root.querySelector("#demo-panel-"+id);
      panel.scrollIntoView({behavior:matchMedia("(prefers-reduced-motion: reduce)").matches?"instant":"smooth",block:"start"});
      if(id==="lidar"){await this.cloud.load();if(this.current===id)this.cloud.reveal();}
      else if(this.evidence.has(id))return;
      else if(id==="camera-lidar")await this.cameraLidar.load();
      else if(id==="camera-imu")await this.cameraImu.load();
      else if(id!=="stereo")this.play(METHODS.find(m=>m.id===id));
    }
  }
  globalThis.CalibrexMethods={METHODS,validateStereo,stereoCrop,MethodGallery};
})();
