// Run this in https://code.earthengine.google.com/
// Exports Andhra Pradesh's historical (2015) district boundaries as GeoJSON.
// These are the boundaries compatible with our 2016-2021 label CSV.

var apPlusTelangana = ee.FeatureCollection('FAO/GAUL/2015/level2')
  .filter(ee.Filter.eq('ADM0_NAME', 'India'))
  .filter(ee.Filter.eq('ADM1_NAME', 'Andhra Pradesh'));

// GAUL's historical Andhra Pradesh group includes the ten Telangana-era
// districts too (23 total). Keep only districts represented in our labels.
// Cuddapah/Kadapa are alternative historical spellings; GAUL contains one.
var targetDistrictNames = [
  'Anantapur', 'Ananthapur', 'Chittoor', 'East Godavari', 'Guntur',
  'Kadapa', 'Cuddapah', 'Y.S.R. Kadapa', 'Ysr Kadapa', 'Krishna', 'Kurnool',
  'Nellore', 'Sri Potti Sriramulu Nellore', 'Spsr Nellore', 'Prakasam',
  'Srikakulam', 'Visakhapatnam', 'Vishakhapatnam', 'Vizianagaram',
  'West Godavari'
];
var apDistricts = apPlusTelangana.filter(
  ee.Filter.inList('ADM2_NAME', targetDistrictNames)
);

print('All returned historical district names:', apPlusTelangana.aggregate_array('ADM2_NAME').sort());
print('Selected district count (should be 13):', apDistricts.size());
print('Selected district names:', apDistricts.aggregate_array('ADM2_NAME').sort());

Map.centerObject(apDistricts, 7);
Map.addLayer(apDistricts, {color: 'yellow'}, 'AP historical districts');

Export.table.toDrive({
  collection: apDistricts,
  description: 'ap_13_district_boundaries',
  folder: 'SIC_exports',
  fileNamePrefix: 'ap_13_districts',
  fileFormat: 'GeoJSON'
});
