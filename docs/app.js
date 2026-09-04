const DATA_URL="https://raw.githubusercontent.com/KarthikNH/Real-estate-price-analysis/main/BHP_3.csv";
const C={Premium:"#C9A96E","High Growth":"#3ECFB2","High Potential":"#6E9EFF",Stable:"#A0AEC0",Budget:"#F6A35A"};
let rows=[], locs=[], summary={}, charts={};

function parseCSV(t){const lines=t.trim().split(/\r?\n/);if(!lines.length)return[];const h=splitCSV(lines[0]);return lines.slice(1).map(l=>{const a=splitCSV(l),o={};h.forEach((k,i)=>o[k]=a[i]??"");return o}).filter(o=>Object.keys(o).length>0)}
function splitCSV(s){let a=[],v="",q=false;for(let i=0;i<s.length;i++){let c=s[i];if(c=='"'){if(q&&s[i+1]=='"'){v+='"';i++}else q=!q}else if(c==','&&!q){a.push(v);v=""}else v+=c}a.push(v);return a}
function num(v){const n=parseFloat(String(v).replace(/[^0-9.-]/g,""));return Number.isFinite(n)?n:0}
function normName(s){return String(s||"").trim()}
function mode(a){const m={};a.forEach(x=>m[x]=(m[x]||0)+1);return Object.entries(m).sort((a,b)=>b[1]-a[1])[0]?.[0]||""}
function mean(a){return a.length?a.reduce((x,y)=>x+y,0)/a.length:0}
function pct(v){return `${v.toFixed(1)}%`}
function money(v){return "₹"+Math.round(v).toLocaleString("en-IN")}
function cat(r){
  const p=num(r.price_per_sqft),g=num(r.annual_growth_rate_pct),conn=connectivity(r);
  if(p>=16000)return"Premium"; if(g>=15)return"High Growth"; if(p<7000)return"Budget";
  if(g>=10&&p<12000&&conn>30)return"High Potential"; return"Stable";
}
function connectivity(r){return num(r.metro_access)*25+Math.max(0,Math.min(20,20-num(r.distance_to_it_hub_km)))+Math.max(0,Math.min(30,30-num(r.distance_to_airport_km)*.5))}
function enrich(r){
  const p21=num(r.price_2021),p25=num(r.price_2025),p23=num(r.price_2023);
  const cagr=p21>0?(Math.pow(p25/p21,.25)-1)*100:0;
  const conn=connectivity(r);
  const roi1=Math.max(0,cagr),roi3=Math.max(0,(Math.pow(1+cagr/100,3)-1)*100),roi5=Math.max(0,(Math.pow(1+cagr/100,5)-1)*100);
  return {...r,_pps:num(r.price_per_sqft),_growth:num(r.annual_growth_rate_pct),_cagr:cagr,_conn:conn,_roi1:roi1,_roi3:roi3,_roi5:roi5,_cat:cat(r),
    _pred1:p25*(1+cagr/100),_pred3:p25*Math.pow(1+cagr/100,3),_pred5:p25*Math.pow(1+cagr/100,5)}
}
function build(){
  rows=rows.map(enrich);
  const by={};rows.forEach(r=>(by[r.location]??=[]).push(r));
  locs=Object.entries(by).map(([location,a])=>({
    location,zone:mode(a.map(x=>x.zone)),count:a.length,lat:mean(a.map(x=>num(x.latitude))),lon:mean(a.map(x=>num(x.longitude))),
    pps:mean(a.map(x=>x._pps)),growth:mean(a.map(x=>x._growth)),roi1:mean(a.map(x=>x._roi1)),roi3:mean(a.map(x=>x._roi3)),roi5:mean(a.map(x=>x._roi5)),
    conn:mean(a.map(x=>x._conn)),score:0,cat:mode(a.map(x=>x._cat)),
    p21:mean(a.map(x=>num(x.price_2021))),p22:mean(a.map(x=>num(x.price_2022))),p23:mean(a.map(x=>num(x.price_2023))),p24:mean(a.map(x=>num(x.price_2024))),p25:mean(a.map(x=>num(x.price_2025)))
  }));
  const mins=k=>Math.min(...locs.map(x=>x[k])),maxs=k=>Math.max(...locs.map(x=>x[k]));
  const n=(v,k)=>{const a=mins(k),b=maxs(k);return b===a?.5:(v-a)/(b-a)};
  locs.forEach(x=>x.score=35*n(x.growth,"growth")+25*n(x.conn,"conn")+25*n(x.roi5,"roi5")+15*(1-n(x.pps,"pps")));
  const best=[...locs].sort((a,b)=>b.score-a.score)[0];
  summary={total:rows.length,locations:locs.length,avgPps:mean(rows.map(r=>r._pps)),avgGrowth:mean(rows.map(r=>r._growth)),best:best?.location||"N/A",bestScore:best?.score||0};
}
function destroy(id){if(charts[id]){charts[id].destroy();delete charts[id]}}
function chart(id,type,data,opts={}){destroy(id);charts[id]=new Chart(document.getElementById(id),{type,data,options:{responsive:true,plugins:{legend:{labels:{color:"#c0cadf"}}},scales:type==="doughnut"?{}:{x:{ticks:{color:"#8a95b0"},grid:{color:"rgba(138,149,176,.12)"}},y:{ticks:{color:"#8a95b0"},grid:{color:"rgba(138,149,176,.12)"}}},...opts}})}
function head(title,sub){return `<div class="page-head"><h1>${title}</h1><p>${sub}</p></div>`}
function footer(){return `<div class="footer">Built by Christo Savio George · Karthik NH · Vamshi MR · Varshith Raj B · Team Iris</div>`}
function renderOverview(){
  const top=[...locs].sort((a,b)=>b.score-a.score).slice(0,10), cats={};rows.forEach(r=>cats[r._cat]=(cats[r._cat]||0)+1);
  document.querySelector("#main").innerHTML=head("Market Overview","Bangalore Property Intelligence · 2021–2025 · GitHub Pages")+
  `<div class="grid kpis">
    ${kpi("Total Records",summary.total.toLocaleString(),"Processed dataset")}
    ${kpi("Unique Locations",summary.locations,"Bangalore micro-markets")}
    ${kpi("Avg Price / sqft",money(summary.avgPps),"Market average")}
    ${kpi("Avg Annual Growth",pct(summary.avgGrowth),"Historical growth")}
    ${kpi("Top City",summary.best,"Highest investment score")}
  </div>
  <div class="grid two"><div class="card"><div class="section">Price Trend · 2021–2025</div><canvas id="trend"></canvas></div>
  <div class="card"><div class="section">Category Distribution</div><canvas id="donut"></canvas></div></div>
  <div class="section">Top Investment Locations</div><div class="grid cards3">${top.slice(0,6).map(locCard).join("")}</div>${footer()}`;
  chart("trend","line",{labels:[2021,2022,2023,2024,2025],datasets:top.slice(0,6).map((x,i)=>({label:x.location,data:[x.p21,x.p22,x.p23,x.p24,x.p25],borderColor:Object.values(C)[i%5],backgroundColor:"transparent",tension:.3}))});
  chart("donut","doughnut",{labels:Object.keys(cats),datasets:[{data:Object.values(cats),backgroundColor:Object.keys(cats).map(k=>C[k])}]},{plugins:{legend:{position:"right"}}});
}
function kpi(a,b,c){return `<div class="card kpi"><div class="label">${a}</div><div class="value">${b}</div><div class="delta">${c}</div></div>`}
function locCard(x){return `<div class="card"><div class="label">${x.zone} Zone</div><h3 style="margin:6px 0">${x.location}</h3><span class="pill" style="color:${C[x.cat]};background:${C[x.cat]}22;border:1px solid ${C[x.cat]}55">${x.cat}</span><div style="margin-top:12px;font-size:12px;color:#c0cadf;display:grid;gap:6px"><span>Price/sqft <b>${money(x.pps)}</b></span><span>Growth <b>${pct(x.growth)}</b></span><span>Score <b style="color:#3ecfb2">${x.score.toFixed(1)}</b></span><span>5Y ROI <b style="color:#6e9eff">${pct(x.roi5)}</b></span></div></div>`}
function renderROI(){
  const top=[...locs].sort((a,b)=>b.roi5-a.roi5).slice(0,15);
  document.querySelector("#main").innerHTML=head("ROI Analysis","1-year, 3-year and 5-year projected returns across Bangalore micro-markets")+
  `<div class="grid two"><div class="card"><div class="section">ROI Forecast · Top 15</div><canvas id="roi"></canvas></div><div class="card"><div class="section">Investment Score vs 5Y ROI</div><canvas id="scatter"></canvas></div></div>
  <div class="section">ROI Data · All Locations</div><div class="card table-wrap"><table class="table"><thead><tr><th>Rank</th><th>Location</th><th>Zone</th><th>Category</th><th>Price/sqft</th><th>ROI 1Y</th><th>ROI 3Y</th><th>ROI 5Y</th><th>Score</th></tr></thead><tbody>${top.map((x,i)=>`<tr><td>${i+1}</td><td>${x.location}</td><td>${x.zone}</td><td>${x.cat}</td><td>${money(x.pps)}</td><td>${pct(x.roi1)}</td><td>${pct(x.roi3)}</td><td>${pct(x.roi5)}</td><td>${x.score.toFixed(1)}</td></tr>`).join("")}</tbody></table></div>${footer()}`;
  chart("roi","bar",{labels:top.map(x=>x.location),datasets:[{label:"1 Year",data:top.map(x=>x.roi1),backgroundColor:"#3ecfb2"},{label:"3 Years",data:top.map(x=>x.roi3),backgroundColor:"#6e9eff"},{label:"5 Years",data:top.map(x=>x.roi5),backgroundColor:"#c9a96e"}]},{indexAxis:"y"});
  chart("scatter","scatter",{datasets:[{label:"Locations",data:locs.map(x=>({x:x.score,y:x.roi5})),backgroundColor:"#3ecfb2"}]},{scales:{x:{title:{display:true,text:"Investment Score",color:"#8a95b0"},ticks:{color:"#8a95b0"},grid:{color:"rgba(138,149,176,.12)"}},y:{title:{display:true,text:"5Y ROI %",color:"#8a95b0"},ticks:{color:"#8a95b0"},grid:{color:"rgba(138,149,176,.12)"}}}});
}
function renderRanking(){
 const top=[...locs].sort((a,b)=>b.score-a.score);
 document.querySelector("#main").innerHTML=head("Investment Ranking","Composite score weighted by growth, connectivity, ROI and value")+
 `<div class="grid two"><div class="card"><div class="section">Investment Score · Top 20</div><canvas id="rankchart"></canvas></div><div><div class="section">Top 10 · Ranked Locations</div>${top.slice(0,10).map((x,i)=>`<div class="rank"><div class="rankno">${String(i+1).padStart(2,"0")}</div><div class="rankmain"><b>${x.location}</b><small>${x.zone} Zone · ${x.cat}</small></div><div class="score">${x.score.toFixed(1)}</div></div>`).join("")}</div></div>${footer()}`;
 chart("rankchart","bar",{labels:top.slice(0,20).map(x=>x.location),datasets:[{label:"Score",data:top.slice(0,20).map(x=>x.score),backgroundColor:top.slice(0,20).map(x=>C[x.cat])}]},{indexAxis:"y",plugins:{legend:{display:false}}});
}
function renderCity(){
 document.querySelector("#main").innerHTML=head("City Deep Dive","Granular analysis per Bangalore micro-market")+
 `<div class="filters"><select id="citySel">${locs.map(x=>`<option>${x.location}</option>`).join("")}</select></div><div id="cityContent"></div>${footer()}`;
 const update=()=>{const x=locs.find(y=>y.location===document.querySelector("#citySel").value)||locs[0];document.querySelector("#cityContent").innerHTML=`<div class="grid cards3">${kpi("Price / sqft",money(x.pps),"2025 market level")}${kpi("Annual Growth",pct(x.growth),"Historical trajectory")}${kpi("Investment Score",x.score.toFixed(1),"Composite score")}</div><div class="grid two"><div class="card"><div class="section">${x.location} · Price History & Forecast</div><canvas id="citychart"></canvas></div><div class="card"><div class="section">Investment Profile</div>${locCard(x)}<div class="insight" style="margin-top:12px">${insight(x)}</div></div></div>`;chart("citychart","line",{labels:[2021,2022,2023,2024,2025,"+1Y","+3Y","+5Y"],datasets:[{label:"Price",data:[x.p21,x.p22,x.p23,x.p24,x.p25,x.p25*(1+x.growth/100),x.p25*Math.pow(1+x.growth/100,3),x.p25*Math.pow(1+x.growth/100,5)],borderColor:"#3ecfb2",tension:.3}]})};document.querySelector("#citySel").onchange=update;update();
}
function insight(x){if(x.cat==="Premium")return`${x.location} commands premium pricing at ${money(x.pps)}/sqft with ${pct(x.growth)} annual growth. 5-year ROI forecast: ${pct(x.roi5)}. Best suited for high-net-worth portfolio diversification.`;if(x.cat==="High Growth")return`${x.location} is among the faster-appreciating micro-markets with ${pct(x.growth)} growth. Investment score: ${x.score.toFixed(0)}/100. Projected 3-year ROI: ${pct(x.roi3)}.`;if(x.cat==="High Potential")return`${x.location} presents an emerging opportunity: accessible pricing, connectivity score ${x.conn.toFixed(0)} and ${pct(x.growth)} growth signal upside. 5-year ROI: ${pct(x.roi5)}.`;if(x.cat==="Budget")return`${x.location} remains affordable at ${money(x.pps)}/sqft, suitable for first-time buyers and rental-oriented investors.`;return`${x.location} offers steady ${pct(x.growth)} annual appreciation. Investment score: ${x.score.toFixed(0)}/100. 5-year ROI estimate: ${pct(x.roi5)}.`}
function renderInsights(){
 const cats=["All","Premium","High Growth","High Potential","Stable","Budget"];
 document.querySelector("#main").innerHTML=head("Market Insights","Location-level intelligence generated from the processed dataset")+
 `<div class="filters"><select id="catSel">${cats.map(x=>`<option>${x}</option>`).join("")}</select></div><div id="insightList"></div>${footer()}`;
 const update=()=>{let a=locs.filter(x=>document.querySelector("#catSel").value==="All"||x.cat===document.querySelector("#catSel").value).sort((a,b)=>b.score-a.score);document.querySelector("#insightList").innerHTML=a.map(x=>`<div class="card" style="margin-bottom:8px;border-left:3px solid ${C[x.cat]}"><div style="display:flex;justify-content:space-between"><b>${x.location}</b><span class="pill" style="color:${C[x.cat]}">${x.cat}</span></div><div class="insight" style="margin-top:9px">${insight(x)}</div></div>`).join("")};document.querySelector("#catSel").onchange=update;update();
}
function renderAgents(){
 const agents=[["01","Data Ingestion Agent","Data Pipeline","Loads and validates the property CSV schema.","raw data","17 required columns"],["02","Data Cleaning Agent","Preprocessing","Removes duplicates, coerces numeric values, fills missing values and filters geographic/outlier records.","clean dataset","validated coordinates"],["03","EDA Agent","Analysis","Computes market statistics, growth rates, zone distributions and location summaries.","EDA summary","YoY growth"],["04","Feature Engineering Agent","ML Preprocessing","Derives price CAGR, momentum, property value and connectivity features.","price CAGR","connectivity score"],["05","Prediction Agent","Machine Learning","Original Python pipeline trains three Gradient Boosting Regressors for 1Y, 3Y and 5Y horizons.","1Y forecast","3Y forecast","5Y forecast"],["06","Investment Analysis Agent","Scoring","Computes ROI, growth index and the weighted 0–100 investment score.","ROI","investment score"],["07","Insight Generation Agent","NLP","Creates structured location-level market narratives and identifies the highest-scoring city.","location insight","best city"],["08","Visualization Prep Agent","Dashboard","Aggregates the processed dataset into location, zone, category and trend artifacts.","location aggregates","rankings"]];
 document.querySelector("#main").innerHTML=head("AI Agent Architecture","8 specialised agents orchestrated in sequence to deliver autonomous real estate intelligence")+
 `<div class="card"><div class="section">Agent Pipeline · Flow</div><div class="grid cards3">${agents.map(a=>`<div class="card agent"><div class="label">Agent ${a[0]} · ${a[2]}</div><h3 style="margin-top:6px">${a[1]}</h3><p>${a[3]}</p><div class="chips">${a.slice(4).map(z=>`<span class="chip">${z}</span>`).join("")}</div></div>`).join("")}</div></div>${footer()}`;
}
function renderData(){
 const cols=["location","zone","property_type","bhk","total_sqft","price_per_sqft","price_2021","price_2025","annual_growth_rate_pct","metro_access"];
 document.querySelector("#main").innerHTML=head("Data Explorer","Processed browser-side view of the GitHub dataset")+
 `<div class="filters"><input id="search" placeholder="Search location…"><select id="rowsN"><option>25</option><option>50</option><option>100</option></select></div><div class="card table-wrap"><table class="table"><thead><tr>${cols.map(c=>`<th>${c}</th>`).join("")}</tr></thead><tbody id="tbody"></tbody></table></div>${footer()}`;
 const update=()=>{let q=document.querySelector("#search").value.toLowerCase(),n=+document.querySelector("#rowsN").value,a=rows.filter(r=>String(r.location).toLowerCase().includes(q)).slice(0,n);document.querySelector("#tbody").innerHTML=a.map(r=>`<tr>${cols.map(c=>`<td>${r[c]??""}</td>`).join("")}</tr>`).join("")};document.querySelector("#search").oninput=update;document.querySelector("#rowsN").onchange=update;update();
}
function renderMap() {
  const mapData = (window.DATA && window.DATA.locations) ? window.DATA.locations : [];
  const C = {
    "Premium": "#C9A96E",
    "High Growth": "#3ECFB2",
    "High Potential": "#6E9EFF",
    "Stable": "#A0AEC0",
    "Budget": "#F6A35A"
  };

  main.innerHTML = `
    <div class="page-head">
      <div>
        <h1>Geospatial Intelligence Map</h1>
        <p>Click any marker for detailed ROI analysis and investment insights.</p>
      </div>
      <select id="mapFilter" class="select">
        <option>All</option>
        <option>Top Premium</option>
        <option>Top Affordable</option>
        <option>High Growth</option>
      </select>
    </div>

    <div class="card map-card">
      <div id="mapCanvas" class="map-canvas"></div>
      <div class="map-legend">
        ${Object.entries(C).map(([k,v]) =>
          `<span><i style="background:${v}"></i>${k}</span>`).join("")}
      </div>
    </div>

    <div class="section-title">Mapped Locations</div>
    <div id="mapCards" class="grid grid-3"></div>
  `;

  const map = L.map("mapCanvas", {
    zoomControl: true,
    scrollWheelZoom: true
  }).setView([12.9716, 77.5946], 11);

  L.tileLayer(
    "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
    }
  ).addTo(map);

  let markers = [];

  function popupHTML(x, color) {
    const money = n => Number(n || 0).toLocaleString("en-IN", {
      maximumFractionDigits: 0
    });

    return `
      <div style="min-width:260px;font-family:Arial,sans-serif">
        <h3 style="margin:0 0 8px;color:${color}">${x.location}</h3>
        <div style="font-size:12px;color:#9aa6bd;margin-bottom:10px">${x.zone || ""}</div>
        <div><b>Category:</b> ${x.cat}</div>
        <div><b>Price/sqft:</b> ₹${money(x.pps)}</div>
        <div><b>Investment Score:</b> ${Number(x.score || 0).toFixed(0)}/100</div>
        <div><b>5-Year ROI:</b> ${Number(x.roi5 || 0).toFixed(1)}%</div>
        <hr style="border:0;border-top:1px solid #33405a;margin:10px 0">
        <div><b>Forecast 1Y:</b> ₹${money(x.p1)}</div>
        <div><b>Forecast 3Y:</b> ₹${money(x.p3)}</div>
        <div><b>Forecast 5Y:</b> ₹${money(x.p5)}</div>
        ${x.insight ? `<p style="margin:10px 0 0;color:#aeb9cf">${x.insight}</p>` : ""}
      </div>
    `;
  }

  function getFiltered(filter) {
    let rows = [...mapData];

    if (filter === "Top Premium") {
      rows = rows.filter(x => x.cat === "Premium")
                   .sort((a,b) => b.score - a.score)
                   .slice(0, 15);
    } else if (filter === "Top Affordable") {
      rows = rows.sort((a,b) => a.pps - b.pps).slice(0, 15);
    } else if (filter === "High Growth") {
      rows = rows.filter(x => x.cat === "High Growth" || x.cat === "High Potential")
                   .sort((a,b) => b.growth - a.growth)
                   .slice(0, 20);
    }

    return rows.filter(x =>
      Number.isFinite(Number(x.lat)) &&
      Number.isFinite(Number(x.lon))
    );
  }

  function draw(filter = "All") {
    markers.forEach(m => map.removeLayer(m));
    markers = [];

    const rows = getFiltered(filter);
    const bounds = [];

    rows.forEach(x => {
      const color = C[x.cat] || "#A0AEC0";
      const marker = L.circleMarker([Number(x.lat), Number(x.lon)], {
        radius: 7 + Math.min(8, Math.max(0, Number(x.score || 0) / 12)),
        color,
        fillColor: color,
        fillOpacity: 0.88,
        weight: 2
      });

      marker.bindPopup(popupHTML(x, color));
      marker.bindTooltip(
        `<b>${x.location}</b><br>${x.cat} · Score ${Number(x.score || 0).toFixed(0)}`,
        {direction: "top", opacity: 0.95}
      );

      marker.addTo(map);
      markers.push(marker);
      bounds.push([Number(x.lat), Number(x.lon)]);
    });

    if (bounds.length === 1) {
      map.setView(bounds[0], 13);
    } else if (bounds.length > 1) {
      map.fitBounds(bounds, {padding: [35, 35], maxZoom: 13});
    } else {
      map.setView([12.9716, 77.5946], 11);
    }

    const cards = document.getElementById("mapCards");
    cards.innerHTML = rows.map(x => {
      const color = C[x.cat] || "#A0AEC0";
      return `
        <div class="card location-card" onclick="window.__openLocation('${String(x.location).replace(/'/g, "\\'")}')">
          <div class="badge" style="border-color:${color};color:${color}">${x.cat}</div>
          <h3>${x.location}</h3>
          <p>${x.zone || "Bangalore"} · ₹${Number(x.pps || 0).toLocaleString("en-IN")}/sqft</p>
          <div class="mini-stats">
            <span>Score <b>${Number(x.score || 0).toFixed(0)}</b></span>
            <span>Growth <b>${Number(x.growth || 0).toFixed(1)}%</b></span>
            <span>ROI 5Y <b>${Number(x.roi5 || 0).toFixed(1)}%</b></span>
          </div>
        </div>
      `;
    }).join("") || `<div class="empty">No locations match this filter.</div>`;
  }

  window.__openLocation = function(name) {
    const idx = mapData.findIndex(x => x.location === name);
    if (idx < 0) return;
    const x = mapData[idx];
    const color = C[x.cat] || "#A0AEC0";
    map.setView([Number(x.lat), Number(x.lon)], 14);
    const marker = markers.find(m => {
      const p = m.getLatLng();
      return Math.abs(p.lat - Number(x.lat)) < 0.00001 &&
             Math.abs(p.lng - Number(x.lon)) < 0.00001;
    });
    if (marker) marker.openPopup();
  };

  document.getElementById("mapFilter").addEventListener("change", e => {
    draw(e.target.value);
  });

  setTimeout(() => map.invalidateSize(), 100);
  draw("All");
}

function render(page){({overview:renderOverview,map:renderMap,roi:renderROI,ranking:renderRanking,city:renderCity,insights:renderInsights,agents:renderAgents,data:renderData}[page]||renderOverview)()}
document.querySelectorAll(".nav").forEach(b=>b.onclick=()=>{document.querySelectorAll(".nav").forEach(x=>x.classList.remove("active"));b.classList.add("active");render(b.dataset.page)});
async function start(){
 try{
  document.querySelector("#load-status").textContent="Fetching BHP_3.csv from GitHub…";
  const r=await fetch(DATA_URL); if(!r.ok)throw new Error("HTTP "+r.status);
  const t=await r.text(); rows=parseCSV(t);
  if(!rows.length)throw new Error("No records found");
  build();document.querySelector("#loading").style.display="none";render("overview");
 }catch(e){
  document.querySelector("#load-status").textContent="Could not fetch CSV. Using a small browser-generated demo dataset.";
  rows=synthetic();build();setTimeout(()=>{document.querySelector("#loading").style.display="none";render("overview")},500);
 }
}
function synthetic(){const L=[["Koramangala",12.935,77.621,"South"],["Whitefield",12.971,77.749,"East"],["Indiranagar",12.978,77.643,"East"],["HSR Layout",12.912,77.640,"South"],["Electronic City",12.845,77.657,"South"],["Hebbal",13.035,77.597,"North"],["Manyata Tech Park",13.049,77.619,"North"],["MG Road",12.975,77.601,"Central"],["Malleshwaram",13.003,77.570,"North"]];return Array.from({length:180},(_,i)=>{const x=L[i%L.length],p=6000+(i*137)%15000,g=5+(i*7)%16,p21=p*.62;return{location:x[0],latitude:x[1],longitude:x[2],zone:x[3],property_type:["Apartment","Villa","Builder Floor"][i%3],bhk:1+i%4,total_sqft:500+(i*83)%2200,price_per_sqft:p,price_2021:p21,price_2022:p21*(1+g/100),price_2023:p21*Math.pow(1+g/100,2),price_2024:p21*Math.pow(1+g/100,3),price_2025:p,annual_growth_rate_pct:g,metro_access:i%2?"Yes":"No",distance_to_it_hub_km:2+(i%18),distance_to_airport_km:10+(i%35)}})}
start();
