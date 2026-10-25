import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import "../docs/app/stereo-cloud.js";
import "../docs/app/stereo-comparison.js";
import "../docs/app/method-gallery.js";
const {METHODS,validateStereo,stereoCrop}=globalThis.CalibrexMethods;
const read=(p)=>JSON.parse(fs.readFileSync(new URL("../docs/app/samples/"+p,import.meta.url),"utf8"));

test("gallery exposes eleven distinct demos and recorded sources remain attributable",()=>{
  assert.equal(METHODS.length,11);assert.equal(new Set(METHODS.map(m=>m.id)).size,11);
  for(const m of METHODS.filter(m=>m.image)) {
    assert.ok(m.source.startsWith("../assets/"));assert.ok(m.license);assert.ok(m.dataset);
    assert.ok(fs.existsSync(new URL("../docs/"+m.image.slice(3),import.meta.url)));
  }
});
test("each native stereo profile has valid images and finite metric receipts",()=>{
  const index=read("fisheye_stereo_demo.json");
  assert.equal(index.schema,"calibrex.fisheye_stereo_demo/v1");
  assert.deepEqual(index.scene.depth_planes_m,[3,6,12]);
  assert.ok(index.method.includes("No truth depths enter matching"));
  for(const code of ["kb4","mei","ds"]){
    const model=validateStereo(read(index.models[code].url));
    assert.ok(model.matches>20);assert.equal(model.baseline_m,.3);
    assert.ok(model.vertical_median_px<.2);assert.ok(model.median_relative_depth_error<.03);
    assert.throws(()=>validateStereo({...model,raw:"javascript:alert(1)"}),/Invalid/);
    assert.throws(()=>validateStereo({...model,vertical_median_px:NaN}),/Invalid/);
    assert.throws(()=>validateStereo({...model,matches:-1}),/Invalid/);
    assert.equal(model.correspondences.length,model.matches);
    const point=model.correspondences[0];
    for(const change of [
      {left:[-1,20]}, {right:[320,20]}, {disparity_px:0},
      {depth_m:point.depth_m+1}, {right:[point.right[0]+1,point.right[1]]},
    ])assert.throws(()=>validateStereo({...model,correspondences:[{...point,...change},...model.correspondences.slice(1)]}),/Invalid/);
    assert.throws(()=>validateStereo({...model,correspondences:[]}),/Invalid/);
  }
});
test("the default fisheye gallery uses real photographs without fabricated depth truth",()=>{
  const index=read("fisheye_real_demo.json");
  assert.equal(index.capture_kind,"real");
  assert.deepEqual(Object.keys(index.frames),["000","014","027"]);
  for(const [frame,receipt] of Object.entries(index.frames)){
    const data=validateStereo(read(receipt.url));
    assert.equal(data.capture_kind,"real");assert.equal(data.frame,frame);
    assert.equal(data.depth_truth,null);assert.equal(data.median_relative_depth_error,undefined);
    assert.deepEqual(data.image_size,[640,400]);assert.ok(data.matches>100);
    assert.ok(data.baseline_m>.099&&data.baseline_m<.100);
    assert.throws(()=>validateStereo({...data,depth:"data:text/html;base64,SGVsbG8="}),/Invalid/);
  }
  const source=fs.readFileSync(new URL("../docs/app/method-gallery.js",import.meta.url),"utf8");
  assert.ok(source.includes('this.json("samples/fisheye_real_demo.json?'));
  assert.ok(!source.includes('this.json("samples/fisheye_stereo_demo.json")'));
});
test("MEI and Double Sphere use their own real-corner fits on the same photograph",()=>{
  const index=read("fisheye_real_demo.json");
  for(const model of ["mei","ds"])assert.deepEqual(index.models[model],["000","014","027"]);
  for(const model of ["mei","ds"])for(const frame of ["000","014","027"]){
    const kb4=read(index.frames[frame].url),data=validateStereo(read(index.model_frames[model][frame].url));
    assert.equal(data.frame,frame);
    assert.equal(data.capture_kind,"real");assert.equal(data.camera_model,model);
    assert.equal(data.raw,kb4.raw);assert.notEqual(data.rectified,kb4.rectified);
    assert.equal(data.depth_truth,null);assert.ok(data.fit_excluded_reprojection_px<1);
    assert.ok(data.matches>100);assert.ok(data.vertical_median_px<1);
  }
});


test("stereo close-ups preserve subpixel centres and never cross a camera boundary",()=>{
  for(const p of [[0,0],[10.25,3.75],[639.75,399.5],[320,200]]){
    const g=stereoCrop(p,640,400);
    assert.equal(g.size,41);
    assert.ok(g.sx>=0&&g.sy>=0&&g.sx+g.sw<=640&&g.sy+g.sh<=400);
    assert.ok(g.dx>=0&&g.dy>=0&&g.dx+g.sw<=41&&g.dy+g.sh<=41);
    assert.equal(g.sx+g.cx-g.dx-.5,p[0]);
    assert.equal(g.sy+g.cy-g.dy-.5,p[1]);
  }
  assert.deepEqual(stereoCrop([0,0],640,400),{sx:0,sy:0,sw:21,sh:21,dx:20,dy:20,cx:20.5,cy:20.5,size:41});
  assert.equal(stereoCrop([639,399],640,400).sw,21);
  for(const p of [[NaN,0],[-1,0],[640,0],[0,400],null,[1]])assert.throws(()=>stereoCrop(p,640,400),/Invalid/);
  assert.throws(()=>stereoCrop([0,0],0,400),/Invalid/);
  // Exercise every measured match, including fractional right-image rows.
  const index=read("fisheye_real_demo.json");
  for(const receipt of [...Object.values(index.frames),...Object.values(index.model_frames).flatMap(Object.values)]){
    const data=read(receipt.url);
    for(const p of data.correspondences)for(const side of ["left","right"]){
      const g=stereoCrop(p[side],...data.image_size);
      assert.ok(Math.abs(g.sx+g.cx-g.dx-.5-p[side][0])<1e-10);
      assert.ok(Math.abs(g.sy+g.cy-g.dy-.5-p[side][1])<1e-10);
    }
  }
});


test("measured stereo cloud unprojects the rectified left camera without changing receipts",()=>{
  const {unproject,bounds,project}=CalibrexStereoCloud;
  const index=read("fisheye_real_demo.json");
  for(const receipt of [...Object.values(index.frames),...Object.values(index.model_frames).flatMap(Object.values)]){
    const data=read(receipt.url),before=JSON.stringify(data),points=unproject(data),[w,h]=data.image_size,f=data.focal_px;
    assert.equal(points.length,data.matches);
    assert.equal(JSON.stringify(data),before);
    assert.ok(points.some(p=>p.z>3)); // far estimates remain, even if a display range hides them
    for(const [i,p] of points.entries()){
      const [x,y,z]=p.xyz,c=data.correspondences[i];
      assert.equal(p.id,i);assert.equal(z,c.depth_m);
      assert.ok(Math.abs(x/z*f+(w-1)/2-c.left[0])<1e-9);
      assert.ok(Math.abs(y/z*f+(h-1)/2-c.left[1])<1e-9);
      assert.ok(Math.abs((x-data.baseline_m)/z*f+(w-1)/2-c.right[0])<1e-9);
    }
    for(const limit of [1,3,Infinity])for(const [yaw,pitch] of [[0,0],[-25,15],[180,75],[-180,-75]]){
      const view={...bounds(points,limit),yaw,pitch,zoom:1};
      for(const p of points.filter(p=>p.z<=limit)){
        const q=project(p.xyz,view);assert.ok(q.x>=0&&q.x<=960&&q.y>=0&&q.y<=520);
      }
    }
  }
  assert.throws(()=>unproject({image_size:[640,400],focal_px:0,correspondences:[]}),/Invalid/);
  assert.throws(()=>unproject({image_size:[640,400],focal_px:250,correspondences:[{left:[20,30],depth_m:-1}]}),/Invalid/);
});

test("stereo cloud uses image-frame axis signs and frontmost picking for overlapping points",()=>{
  const {project,pick}=CalibrexStereoCloud,view={center:[0,0,0],radius:1,yaw:0,pitch:0,zoom:1};
  assert.deepEqual(project([0,0,0],view),{x:480,y:260,depth:0});
  assert.ok(project([1,0,0],view).x>480);assert.ok(project([0,1,0],view).y>260);
  assert.equal(project([0,0,1],view).depth,1);
  assert.ok(project([0,0,1],{...view,yaw:90}).x>480);
  const points=[{id:3,x:100,y:100,depth:2},{id:7,x:100,y:100,depth:1}];
  assert.equal(pick(points,100,100,12),7);
  assert.equal(pick(points.slice().reverse(),100,100,12),7);
  assert.equal(pick(points,140,140,12),null);
  assert.equal(pick(points,100,110,12),7);
});


test("parallel model diagnostics preserve every match with common bins and model-specific denominators",()=>{
  const {mismatchProfile,comparisonProfiles}=CalibrexStereoComparison,index=read("fisheye_real_demo.json");
  for(const code of ["000","014","027"]){
    const data=[read(index.frames[code].url),read(index.model_frames.mei[code].url),read(index.model_frames.ds[code].url)],before=JSON.stringify(data);
    const profiles=comparisonProfiles(data,code);
    for(const [i,p] of profiles.entries()){
      assert.equal(p.n,data[i].matches);
      assert.equal(p.counts.reduce((s,n)=>s+n,0)+p.overflow,p.n);
      assert.ok(Math.abs(p.percentages.reduce((s,n)=>s+n,0)+p.overflow_pct-100)<1e-9);
      assert.ok(Math.abs(p.median_px-data[i].vertical_median_px)<1e-9);
    }
    assert.equal(JSON.stringify(data),before);
    for(const change of [{frame:"000"===code?"014":"000"},{raw:data[0].raw+"x"},{focal_px:data[0].focal_px+1},{camera_model:"kb4"},{image_size:[320,200]}]){
      assert.throws(()=>comparisonProfiles([data[0],{...data[1],...change},data[2]],code),/Mismatched/);
    }
    assert.throws(()=>comparisonProfiles(data.slice(1),code),/three/);
  }
  const p=mismatchProfile({matches:4,correspondences:[0,.25,-2.99,3].map(v=>({left:[0,0],right:[0,v]}))});
  assert.equal(p.counts[0],1);assert.equal(p.counts[1],1);assert.equal(p.counts[11],1);assert.equal(p.overflow,1);assert.equal(p.overflow_pct,25);
  assert.throws(()=>mismatchProfile({matches:0,correspondences:[]}),/Invalid/);
  assert.throws(()=>mismatchProfile({matches:1,correspondences:[{left:[0,NaN],right:[0,0]}]}),/Invalid/);
});
