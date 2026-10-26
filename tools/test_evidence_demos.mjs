import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import "../docs/app/demo-solver.js";
import "../docs/app/board-residuals.js";
import "../docs/app/evidence-demos.js";
const read=code=>JSON.parse(fs.readFileSync(new URL(`../docs/app/samples/evidence_${code}.json`,import.meta.url),"utf8"));
const {validate}=globalThis.CalibrexEvidence;
const {LiveFit}=globalThis.CalibrexDemoSolver;

test("all five evidence datasets use finite measured coordinates or channels",()=>{
  for(const code of ["comparison","handeye","gnss","vehicle","radar"]){
    const data=validate(read(code),code);
    assert.throws(()=>validate({...data,capture_kind:"synthetic"},code),/Invalid/);
    const first=data.views[0];
    if(first.kind!=="lines")assert.throws(()=>validate({...data,views:[{...first,clouds:[{points:[[NaN,0,1]]}]}]},code),/Invalid/);
    else assert.throws(()=>validate({...data,views:[{...first,chart:{...first.chart,x:[1]}}]},code),/Invalid/);
  }
  assert.equal(read("comparison").views.length,4);
  assert.equal(read("handeye").views.length,8);
  assert.equal(read("vehicle").views.length,5);
});
test("live 6-DoF fit performs real updates and never fits held-out centres",()=>{
  const problem=read("comparison").live_problem;
  const fit=new LiveFit(problem),initial=fit.history[0].fit;
  while(fit.step()){}
  assert.equal(fit.status,"converged");assert.ok(fit.iteration>=2);
  assert.ok(fit.history.at(-1).fit<initial/10);
  assert.ok(fit.history.at(-1).holdout<.03);
  for(let i=1;i<fit.history.length;i++)assert.ok(fit.history[i].fit<=fit.history[i-1].fit+1e-12);
  const changed=structuredClone(problem);
  changed.ids.forEach((id,i)=>{if(!problem.fit_ids.includes(id))changed.target[i]=changed.target[i].map(v=>v+100);});
  const other=new LiveFit(changed);while(other.step()){}
  fit.state.forEach((v,i)=>assert.ok(Math.abs(v-other.state[i])<1e-10));
  assert.ok(other.history.at(-1).holdout>100);
});
test("limited measured coverage is blocked before damping can hide a missing axis",()=>{
  const problem=read("comparison").live_problem,fit=new LiveFit(problem,{scenario:"limited"});
  assert.equal(fit.fitIndices.length,2);assert.equal(fit.holdIndices.length,8);
  assert.equal(fit.geometry.rank,5);assert.equal(fit.geometry.rotationRatio,0);
  const initial=[...fit.state];assert.equal(fit.step(),false);
  assert.equal(fit.status,"underconstrained");assert.equal(fit.iteration,0);
  assert.deepEqual(fit.state,initial);assert.equal(fit.history.length,1);
  const g=globalThis.CalibrexDemoSolver.geometry;
  assert.equal(g([[0,0,0],[1,0,0],[0,1,0]]).rank,6);
  assert.equal(g([[0,0,0],[1,0,0],[2,0,0]]).rank,5);
  assert.equal(g([[1,1,1],[1,1,1]]).rank,3);
});
test("outlier controls preserve sources and holdouts; Huber reduces clean holdout damage",()=>{
  const problem=read("comparison").live_problem,original=structuredClone(problem);
  const fits=["least_squares","huber"].map(loss=>new LiveFit(problem,{scenario:"outliers",loss}));
  for(const fit of fits){
    assert.equal(fit.injectedIndices.length,4);
    for(const i of fit.holdIndices)assert.deepEqual(fit.targets[i],problem.target[i]);
    assert.ok(fit.injectedIndices.every(i=>fit.fitIndices.includes(i)));
    while(fit.step()){}
    assert.equal(fit.status,"converged");
    for(let i=1;i<fit.history.length;i++)assert.ok(fit.history[i].objective<=fit.history[i-1].objective+1e-12);
  }
  assert.ok(fits[0].history.at(-1).holdout>.08);
  assert.ok(fits[1].history.at(-1).holdout<.02);
  assert.ok(fits[1].history.at(-1).holdout<fits[0].history.at(-1).holdout/5);
  assert.deepEqual(problem,original);
  const altered=structuredClone(problem);
  altered.ids.forEach((id,i)=>{if(!altered.fit_ids.includes(id))altered.target[i]=[100,100,100];});
  const robust=new LiveFit(altered,{scenario:"outliers",loss:"huber"});while(robust.step()){}
  assert.deepEqual(robust.state,fits[1].state);
  assert.throws(()=>new LiveFit(problem,{scenario:"unknown"}),/Unknown/);
});


test("board matrix binds shared held-out capture IDs and agrees with native full-precision RMSE",()=>{
  const {residualMatrix}=CalibrexBoardResiduals,data=read("comparison"),before=JSON.stringify(data),m=residualMatrix(data);
  assert.equal(m.rows.length,4);assert.equal(m.ids.length,8);assert.equal(m.fitCount,32);
  assert.deepEqual(m.ids,data.views[0].result.holdout_frame_ids);
  for(const [i,row] of m.rows.entries()){
    // Each displayed coordinate was rounded to 1e-7 m: allow <0.0002 mm
    // for the norm error across the two rounded 3D endpoints.
    assert.ok(Math.abs(row.rmseMm-data.views[i].metrics[0].value*1000)<.0002);
    assert.equal(row.cells.length,8);
    row.cells.forEach((cell,j)=>{assert.equal(cell.id,m.ids[j]);assert.ok(Number.isFinite(cell.valueMm)&&cell.valueMm>=0);});
  }
  assert.equal(JSON.stringify(data),before);
  const shifted=structuredClone(data),offset=[3,-8,12];
  for(const view of shifted.views)for(const cloud of view.clouds)cloud.points=cloud.points.map(p=>p.map((v,k)=>v+offset[k]));
  shifted.live_problem.target=shifted.live_problem.target.map(p=>p.map((v,k)=>v+offset[k]));
  const other=residualMatrix(shifted);
  m.rows.forEach((row,i)=>row.cells.forEach((c,j)=>assert.ok(Math.abs(c.valueMm-other.rows[i].cells[j].valueMm)<1e-8)));
  const known=structuredClone(data),camera=known.views[0].clouds[0].points[0];
  known.views[0].clouds[1].points[0]=camera.map((v,k)=>v+[.003,.004,0][k]);
  assert.ok(Math.abs(residualMatrix(known).rows[0].cells[0].valueMm-5)<1e-9);
});

test("board matrix rejects reused fit captures, mismatched methods and mislabelled observations",()=>{
  const {residualMatrix}=CalibrexBoardResiduals,original=read("comparison");
  const changes=[
    d=>d.views[1].result.holdout_frame_ids.reverse(),
    d=>d.views[1].result.train_frame_ids.pop(),
    d=>d.views[0].result.train_frame_ids.push(d.views[0].result.holdout_frame_ids[0]),
    d=>d.live_problem.fit_ids.push(d.views[0].result.holdout_frame_ids[0]),
    d=>d.views[1].clouds[0].points.reverse(),
    d=>d.views[1].clouds[1].points.pop(),
    d=>d.views[1].clouds[1].points[0][0]=NaN,
    d=>d.views[1].id=d.views[0].id,
    d=>d.views[1].id="",
    d=>{for(const v of d.views){v.result.holdout_frame_ids.pop();for(const c of v.clouds)c.points.pop();}},
  ];
  for(const apply of changes){const data=structuredClone(original);apply(data);assert.throws(()=>residualMatrix(data),/Invalid|Mismatched|Capture/);}
  assert.throws(()=>residualMatrix(read("handeye")),/Invalid/);
});
