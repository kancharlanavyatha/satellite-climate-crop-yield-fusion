// Run this in https://code.earthengine.google.com/ after 01_export_ap_boundaries.js.
// Creates two Google Drive downloads: district_static_features.csv and
// district_year_weather.csv. No separate weather, soil, or DEM download is needed.

var rawDistricts = ee.FeatureCollection('FAO/GAUL/2015/level2')
  .filter(ee.Filter.eq('ADM0_NAME', 'India'))
  .filter(ee.Filter.eq('ADM1_NAME', 'Andhra Pradesh'));

var targetNames = [
  'Anantapur', 'Ananthapur', 'Chittoor', 'East Godavari', 'Guntur',
  'Kadapa', 'Cuddapah', 'Y.S.R. Kadapa', 'Ysr Kadapa', 'Krishna', 'Kurnool',
  'Nellore', 'Sri Potti Sriramulu Nellore', 'Spsr Nellore', 'Prakasam',
  'Srikakulam', 'Visakhapatnam', 'Vishakhapatnam', 'Vizianagaram',
  'West Godavari'
];
var nameMap = ee.Dictionary({
  'Anantapur': 'Anantapur', 'Ananthapur': 'Anantapur',
  'Cuddapah': 'Kadapa', 'Kadapa': 'Kadapa', 'Y.S.R. Kadapa': 'Kadapa',
  'Ysr Kadapa': 'Kadapa', 'Nellore': 'Nellore',
  'Sri Potti Sriramulu Nellore': 'Nellore', 'Spsr Nellore': 'Nellore'
});
var districts = rawDistricts.filter(ee.Filter.inList('ADM2_NAME', targetNames))
  .map(function(feature) {
    var original = ee.String(feature.get('ADM2_NAME'));
    return feature.set('District', nameMap.get(original, original));
  });

print('Selected districts (must be 13):', districts.size());

function meanByDistrict(image, scale) {
  return districts.map(function(district) {
    var values = image.reduceRegion({
      reducer: ee.Reducer.mean(), geometry: district.geometry(), scale: scale,
      maxPixels: 1e10, bestEffort: true
    });
    return ee.Feature(null, values).set('District', district.get('District'));
  });
}

// STATIC FEATURES: elevation, slope, and topsoil properties.
var elevation = ee.Image('USGS/SRTMGL1_003').select('elevation').rename('Elevation_m');
var slope = ee.Terrain.slope(elevation).rename('Slope_deg');
var clay = ee.Image('OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02')
  .select('b0').rename('Clay_pct');
var sand = ee.Image('OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02')
  .select('b0').rename('Sand_pct');
var organicCarbon = ee.Image('OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02')
  .select('b0').multiply(5).rename('OrganicCarbon_g_kg');
var soilPH = ee.Image('OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02')
  .select('b0').multiply(0.1).rename('Soil_pH');
var staticImage = elevation.addBands(slope).addBands(clay).addBands(sand)
  .addBands(organicCarbon).addBands(soilPH);
var staticFeatures = meanByDistrict(staticImage, 250);

Export.table.toDrive({
  collection: staticFeatures, description: 'district_static_features',
  folder: 'SIC_exports', fileNamePrefix: 'district_static_features', fileFormat: 'CSV'
});

// YEARLY KHARIF WEATHER FEATURES: June 1 to October 31 for 2017-2021.
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').select('precipitation');
var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR');
var years = [2017, 2018, 2019, 2020, 2021];
var weatherFeatures = ee.FeatureCollection([]);

years.forEach(function(year) {
  var start = ee.Date.fromYMD(year, 6, 1);
  var aug1 = ee.Date.fromYMD(year, 8, 1);
  var oct1 = ee.Date.fromYMD(year, 10, 1);
  var end = ee.Date.fromYMD(year, 11, 1);
  var rainEarly = chirps.filterDate(start, aug1).sum().rename('Rain_Early_mm');
  var rainMid = chirps.filterDate(aug1, oct1).sum().rename('Rain_Mid_mm');
  var rainLate = chirps.filterDate(oct1, end).sum().rename('Rain_Late_mm');
  var rainTotal = chirps.filterDate(start, end).sum().rename('Rain_Total_mm');
  var temperature = era5.filterDate(start, end).select('temperature_2m')
    .mean().subtract(273.15).rename('Temp_Mean_C');
  var heatDays = era5.filterDate(start, end).select('temperature_2m_max')
    .map(function(image) { return image.gte(308.15); }).sum().rename('HeatStressDays_gt35C');
  var annualImage = rainEarly.addBands(rainMid).addBands(rainLate).addBands(rainTotal)
    .addBands(temperature).addBands(heatDays);
  var annualFeatures = meanByDistrict(annualImage, 5500).map(function(feature) {
    return feature.set('Year', year).set('Season', 'Kharif');
  });
  weatherFeatures = weatherFeatures.merge(annualFeatures);
});

Export.table.toDrive({
  collection: weatherFeatures, description: 'district_year_weather',
  folder: 'SIC_exports', fileNamePrefix: 'district_year_weather', fileFormat: 'CSV'
});
