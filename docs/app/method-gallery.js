"use strict";
(() => {
  const METHODS = [
    {id:"lidar",label:"LiDAR ↔ LiDAR",tag:"Point clouds"},
    {id:"deskew",label:"IMU ↔ LiDAR",tag:"Motion deskew",title:"Straighten a moving sweep.",
      description:"One MID360 sweep during a 180°/s turn. The IMU-LiDAR estimate compensates rotation using each point’s measurement time; translation is not compensated.",
      image:"../assets/mid360-deskew.gif",source:"../assets/mid360-deskew/manifest.json",dataset:"RTK-SLAM · construction_seq1",license:"CC BY 4.0",
      metric:"Local surface thickness · proxy, not accuracy",unit:"cm",before:"Raw sweep",after:"Gyro deskewed"},
    {id:"camera-imu",label:"Camera ↔ IMU",tag:"Angular-rate alignment",title:"Make the rate curves agree.",
      description:"Camera-derived angular rates and gyro rates in the camera frame, on Hilti development data. Rotation, clock offset and bias move between a perturbed hypothesis and the saved estimate.",
      image:"../assets/camera-imu-lock.gif",source:"../assets/camera_imu_lock/series.provenance.json",dataset:"Hilti 2022 · exp21 / cam0",license:"Hilti SLAM Challenge 2022 terms",
      metric:"Displayed-interval angular-rate RMSE",unit:"deg/s",before:"Perturbed hypothesis",after:"Saved estimate"},
    {id:"camera-lidar",label:"Camera ↔ LiDAR",tag:"Image projection",title:"Put the points on the objects.",
      description:"Two real KITTI camera-LiDAR checks: a +3° injected camera rotation and the vendor transform. Only roll and pitch are judged here; translation remains unchecked.",
      image:"../assets/calibrex-check-camera-lidar.gif",source:"../assets/calibrex_check_camera_lidar/provenance.json",dataset:"KITTI · drive 0005",license:"CC BY-NC-SA 3.0",
      metric:"Pitch error against the estimate",unit:"deg",before:"Injected rotation",after:"Vendor transform"},
    {id:"stereo",label:"Fisheye stereo",tag:"KB4 / MEI / DS"},
  ];
  const validPNG = (value) => typeof value==="string" && value.length<1800000 && /^data:image\/png;base64,[A-Za-z0-9+/=]+$/.test(value);
  function validateStereo(data){
    if(!data||!Number.isInteger(data.matches)||data.matches<0||![data.vertical_median_px,data.median_relative_depth_error,data.baseline_m,data.focal_px].every(v=>Number.isFinite(v)&&v>=0)||data.baseline_m<=0||data.focal_px<=0||![data.raw,data.rectified,data.depth].every(validPNG))throw Error("Invalid stereo demo data");
    return data;
  }
  class MethodGallery {
    constructor(root,cloud){
      this.root=root;this.cloud=cloud;this.current="lidar";this.request=0;this.modelRequest=0;this.cache=new Map();
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
      if(method.id==="stereo"){
        panel.innerHTML=`<div class="method-media"><div class="cloud-topline"><span>SYNTHETIC GEOMETRY / NATIVE RECTIFICATION</span><span>0.30 m baseline</span></div><div class="stereo-picker"><label>Camera model <select aria-label="Fisheye stereo camera model"><option value="kb4">Equidistant / KB4</option><option value="mei">Unified / MEI + radtan</option><option value="ds">Double Sphere</option></select></label></div><div class="stereo-views"><figure><figcaption>RAW · LEFT / RIGHT</figcaption><img class="stereo-raw" alt="Synthetic raw left and right fisheye images"></figure><figure><figcaption>RECTIFIED · EPIPOLAR GUIDES</figcaption><img class="stereo-rectified" alt="Native rectified images with horizontal guides"></figure><figure class="stereo-depth-figure"><figcaption>SPARSE MATCHED DEPTH · NEAR <span class="depth-key"></span> FAR</figcaption><div class="stereo-depth-row"><img class="stereo-depth" alt="Depth from image patch matches; colored sparse samples"><p>Each dot is an accepted image patch match.<br><br>Its horizontal disparity gives metric depth through the known focal length and baseline.<br><br>Orange: near 3 m<br>Cyan: near 12 m</p></div></figure></div><p class="method-status" role="status">Loading model demo…</p></div><aside class="cloud-evidence"><div class="eyebrow">FISHEYE STEREO / MODEL CHECK</div><h2>One pair.<br>Three lens models.</h2><p class="cloud-description">Native rectification and sparse ZNCC matching for equidistant KB4, unified MEI and Double Sphere. Known scene planes sit at 3, 6 and 12 m.</p><div class="stereo-stats"></div><p class="cloud-description">Synthetic validation, not a real-camera accuracy claim. Integer disparity and matching outliers are retained. Depth is computed as Z = fB/d.</p><a class="method-link" href="../tutorials/fisheye_stereo/">Models, API &amp; limitations ↗</a><a class="method-link" href="samples/fisheye_stereo_demo.json">Generator &amp; receipts ↗</a></aside>`;
        panel.querySelector("select").addEventListener("change",(e)=>this.loadModel(e.target.value));
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
      this.request++;const request=this.request;this.current=id;this.cloud.stop();
      for(const panel of this.panels.children){panel.hidden=panel.id!=="demo-panel-"+id && !(id==="lidar"&&panel.id==="alignment-demo");if(panel.hidden&&panel.querySelector("[data-replay]"))this.pause(panel);}
      for(const button of this.tabs.children){const selected=button.dataset.method===id;button.setAttribute("aria-selected",String(selected));button.tabIndex=selected?0:-1;}
      const panel=id==="lidar"?this.root.querySelector("#alignment-demo"):this.root.querySelector("#demo-panel-"+id);
      panel.setAttribute("aria-labelledby","demo-tab-"+id);panel.role="tabpanel";
      if(id==="lidar")return;
      if(id==="stereo"){this.loadModel(panel.querySelector("select").value);return;}
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
    async loadModel(code){
      const panel=this.root.querySelector("#demo-panel-stereo"),seq=++this.modelRequest,status=panel.querySelector(".method-status");status.textContent="Loading model demo…";
      panel.querySelector(".stereo-views").hidden=true;
      try{
        const index=await this.json("samples/fisheye_stereo_demo.json");
        if(!["kb4","mei","ds"].includes(code)||index.schema!=="calibrex.fisheye_stereo_demo/v1"||!index.models[code])throw Error("Unknown stereo model");
        const data=validateStereo(await this.json("samples/fisheye_stereo_"+code+".json"));if(seq!==this.modelRequest)return;
        for(const key of ["raw","rectified","depth"])panel.querySelector(".stereo-"+key).src=data[key];
        panel.querySelector(".stereo-stats").textContent=`${data.matches} image matches\nMedian vertical mismatch: ${data.vertical_median_px.toFixed(3)} px\nMedian depth error against scene truth: ${(100*data.median_relative_depth_error).toFixed(1)}%`;
        panel.querySelector(".stereo-views").hidden=false;status.textContent="Native synthetic model check · raw, rectified and matched depth";
      }catch(error){if(seq===this.modelRequest)status.textContent=error.message;}
    }
    async explore(){
      const id=this.current,panel=id==="lidar"?this.root.querySelector("#alignment-demo"):this.root.querySelector("#demo-panel-"+id);
      panel.scrollIntoView({behavior:matchMedia("(prefers-reduced-motion: reduce)").matches?"instant":"smooth",block:"start"});
      if(id==="lidar"){await this.cloud.load();if(this.current===id)this.cloud.reveal();}
      else if(id!=="stereo")this.play(METHODS.find(m=>m.id===id));
    }
  }
  globalThis.CalibrexMethods={METHODS,validateStereo,MethodGallery};
})();
