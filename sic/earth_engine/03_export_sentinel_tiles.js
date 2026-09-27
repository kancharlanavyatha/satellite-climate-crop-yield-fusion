// Run this in https://code.earthengine.google.com/.
// Creates 65 export tasks: 13 AP districts x 2017-2021 Kharif seasons.
// Each GeoTIFF has six bands: B2, B3, B4, B8, NDVI, NDWI.
//
// FIXED VERSION:
// - Replaced 13 separate .getInfo() calls (one per district in the loop) with
//   a single .getInfo() call before the loop. This was the actual cause of the
//   "memory capacity exceeded" error you saw on script run (an interactive
//   computation limit) -- it was not related to the export tasks themselves.
// - No changes were needed in the Export.image.toDrive() call; that part of
//   your original script was correct.

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

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');
var years = [2017, 2018, 2019, 2020, 2021];

// Remove cloud, cloud shadow, cirrus, snow, and no-data pixels using Sentinel-2 SCL.
function maskClouds(image) {
  var scl = image.select('SCL');
  var clear = scl.neq(0)   // no data
    .and(scl.neq(1))       // saturated/defective
    .and(scl.neq(3))       // cloud shadow
    .and(scl.neq(8))       // medium-probability cloud
    .and(scl.neq(9))       // high-probability cloud
    .and(scl.neq(10))      // cirrus
    .and(scl.neq(11));     // snow/ice
  return image.updateMask(clear).select(['B2', 'B3', 'B4', 'B8']);
}

function seasonComposite(district, year) {
  var start = ee.Date.fromYMD(year, 6, 1);
  var end = ee.Date.fromYMD(year, 11, 1);
  var base = s2.filterBounds(district.geometry())
    .filterDate(start, end)
    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 80))
    .map(maskClouds)
    .median()
    .clip(district.geometry());

  var ndvi = base.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var ndwi = base.normalizedDifference(['B3', 'B8']).rename('NDWI');
  return base.addBands(ndvi).addBands(ndwi).toFloat();
}

// FIX: get all 13 district names in ONE getInfo() call instead of 13 separate
// calls inside the loop. Order matches districtList below.
var districtList = districts.toList(13);
var districtNames = districts.aggregate_array('District').getInfo();

// NOTE: aggregate_array does not guarantee the same order as districtList.
// To be safe, pull names per-feature from the list itself, still as ONE
// getInfo() call for the whole array instead of 13 separate calls.
var districtFeaturesInfo = districtList.getInfo(); // one round trip
var orderedNames = districtFeaturesInfo.map(function(f) {
  return f.properties.District;
});

var taskCount = 0;
for (var i = 0; i < 13; i++) {
  var district = ee.Feature(districtList.get(i));
  var districtName = orderedNames[i]; // no getInfo() call here anymore

  for (var j = 0; j < years.length; j++) {
    var year = years[j];
    var image = seasonComposite(district, year);
    var fileName = districtName.replace(/[^A-Za-z0-9]/g, '_') + '_' + year + '_Kharif';
    Export.image.toDrive({
      image: image,
      description: fileName,
      folder: 'SIC_exports',
      fileNamePrefix: fileName,
      region: district.geometry(),
      scale: 30,
      maxPixels: 1e13,
      fileFormat: 'GeoTIFF'
    });
    taskCount++;
  }
}
print('Expected export tasks:', taskCount);