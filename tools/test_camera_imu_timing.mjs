import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import "../docs/app/camera-imu-timing.js";
const {validate,evaluate,cumulativeAt,rotateEvaluation,surface,surfacePoint,makeWindows,subset}=globalThis.CalibrexTiming;
const read=()=>JSON.parse(fs.readFileSync(new URL("../docs/app/samples/camera_imu_timing.json",import.meta.url),"utf8"));

test("clock shifts match independent existing Python interval metrics on the same real samples",()=>{
  const data=validate(read()),before=JSON.stringify(data);
  assert.equal(data.camera_start_s.length,59);
  for(const ref of data.references){
    const result=evaluate(data,ref.shift_ms);
    assert.equal(result.predicted.length,59);
    assert.ok(Math.abs(result.rmse_dps-ref.rmse_dps)<1e-9);
    result.predicted.forEach((p,i)=>p.forEach((v,k)=>assert.ok(Math.abs(v-ref.predicted_dps[i][k])<1e-9)));
    assert.ok(Math.abs(result.rmse_dps**2-result.axis_rmse_dps.reduce((s,v)=>s+v*v,0))<1e-9);
    assert.equal(result.offset_s,data.saved_offset_s+ref.shift_ms/1000);
  }
  assert.ok(evaluate(data,50).rmse_dps>evaluate(data,0).rmse_dps*2);
  assert.ok(evaluate(data,-50).rmse_dps>evaluate(data,0).rmse_dps*2);
  assert.equal(JSON.stringify(data),before);
  assert.throws(()=>evaluate(data,51),/within/);
  assert.throws(()=>evaluate(data,NaN),/within/);
});
test("sampled cumulative interpolation does not silently clamp unsupported times",()=>{
  const times=[0,1,2],values=[[0,0,0],[1,2,3],[4,8,12]];
  assert.deepEqual(cumulativeAt(.5,times,values),[.5,1,1.5]);
  assert.deepEqual(cumulativeAt(2,times,values),[4,8,12]);
  assert.throws(()=>cumulativeAt(-.01,times,values),/support/);
  assert.throws(()=>cumulativeAt(2.01,times,values),/support/);
  const d=read();
  assert.throws(()=>validate({...d,imu_time_s:[...d.imu_time_s].reverse()}),/Invalid/);
  assert.throws(()=>validate({...d,camera_start_s:[-100,...d.camera_start_s.slice(1)]}),/extrapolation/);
  assert.throws(()=>validate({...d,camera_rate_dps:[[Infinity,0,0],...d.camera_rate_dps.slice(1)]}),/Invalid/);
});
test("camera-frame rotation and clock slices match native Python rotation order",()=>{
  const data=validate(read()),before=JSON.stringify(data),profiles=Array.from({length:101},(_,i)=>evaluate(data,i-50));
  for(const ref of data.rotation_references){
    const result=rotateEvaluation(data,profiles[ref.shift_ms+50],ref.degrees,ref.axis);
    assert.ok(Math.abs(result.rmse_dps-ref.rmse_dps)<1e-9);
  }
  for(const axis of [0,1,2]){
    const map=surface(data,profiles,axis);
    assert.equal(map.rows.length,41);assert.equal(map.rows[0].length,101);
    for(const i of [0,10,20,30,40])for(const j of [0,25,50,75,100])assert.ok(Math.abs(map.rows[i][j]-rotateEvaluation(data,profiles[j],map.angles[i],axis).rmse_dps)<1e-9);
    assert.ok(Math.abs(map.rows[20][50]-evaluate(data,0).rmse_dps)<1e-9);
    assert.equal(map.minimum.rmse_dps,Math.min(...map.rows.flat()));
    assert.equal(map.similar_count,map.rows.flat().filter(v=>v<=map.minimum.rmse_dps+.5).length);
  }
  assert.equal(JSON.stringify(data),before);
  assert.throws(()=>rotateEvaluation(data,profiles[0],6,0),/Rotation/);
  assert.throws(()=>rotateEvaluation(data,profiles[0],1,3),/Rotation/);
  assert.deepEqual(surfacePoint(505,191),{shift_ms:0,rotation_deg:0});
  assert.deepEqual(surfacePoint(70,26),{shift_ms:-50,rotation_deg:5});
  assert.deepEqual(surfacePoint(940,356),{shift_ms:50,rotation_deg:-5});
  assert.equal(surfacePoint(20,200),null);
});
test("motion windows use camera rates only, equal counts, and disjoint measured observations",()=>{
  const data=read(),before=JSON.stringify(data),windows=makeWindows(data);
  for(const [i,w] of windows.entries()){
    assert.deepEqual(w.indices,data.motion_windows[i].indices);
    assert.ok(Math.abs(w.camera_rms_dps-data.motion_windows[i].camera_rms_dps)<1e-10);
    assert.equal(w.start_s,data.camera_start_s[w.indices[0]]);
    assert.equal(w.end_s,data.camera_end_s[w.indices.at(-1)]);
  }
  const [all,high,low]=windows;
  assert.equal(all.indices.length,59);assert.equal(high.indices.length,20);assert.equal(low.indices.length,20);
  assert.ok(high.camera_rms_dps>low.camera_rms_dps);
  assert.ok(!high.indices.some(i=>low.indices.includes(i)));
  const changed=structuredClone(data);changed.saved_offset_s=100;changed.gyro_cumulative_camera_deg=changed.gyro_cumulative_camera_deg.map(()=>[100,200,300]);
  assert.deepEqual(makeWindows(changed),windows);
  for(const w of [high,low]){
    const selected=validate(subset(data,w.indices));
    for(const shift of [-50,0,50])assert.equal(evaluate(selected,shift).predicted.length,20);
    assert.ok(selected.camera_start_s.every((v,i)=>v===data.camera_start_s[w.indices[i]]));
  }
  assert.equal(JSON.stringify(data),before);
  assert.throws(()=>subset(data,[1,1]),/indices/);
  assert.throws(()=>subset(data,[59]),/indices/);
  assert.throws(()=>makeWindows(data,30),/disjoint/);
});
test("every selected-window slice matches original-frame Python reference metrics",()=>{
  const data=read(),windows=makeWindows(data);
  for(const ref of data.window_references){
    const w=windows.find(w=>w.id===ref.window),selected=subset(data,w.indices);
    const result=rotateEvaluation(selected,evaluate(selected,ref.shift_ms),ref.degrees,ref.axis);
    assert.ok(Math.abs(result.rmse_dps-ref.rmse_dps)<1e-9);
  }
  for(const w of windows){
    const selected=subset(data,w.indices),profiles=Array.from({length:101},(_,i)=>evaluate(selected,i-50));
    for(const axis of [0,1,2]){
      const grid=surface(selected,profiles,axis);
      for(const i of [0,20,40])for(const j of [0,50,100])assert.ok(Math.abs(grid.rows[i][j]-rotateEvaluation(selected,profiles[j],grid.angles[i],axis).rmse_dps)<1e-9);
    }
  }
});
