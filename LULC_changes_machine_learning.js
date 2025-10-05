// Establish area of interest - aoi
var aoi = ee.FeatureCollection('WCMC/WDPA/current/polygons')
   .filter(ee.Filter.inList('NAME', ['Reserva De Desenvolvimento Sustentável Do Rio Negro']));
var visParams = {
  palette: ['2ed033', '5aff05', '67b9ff', '5844ff', '0a7618', '2c05ff'], min: 0.0, max: 1550000.0, opacity: 0.8,
}; 
Map.addLayer(aoi, visParams, 'WCMC/WDPA polygons', false);

var startDate1 = '2020-04-01';
var endDate1 = '2020-08-31';
var startDate2 = '2025-04-01';
var endDate2 = '2025-08-31';

// Define constants
var QA_BAND = 'cs';               // band name
var CLEAR_THRESHOLD = 0.60;       // threshold (0=no cloud, 1=cloud)

// Sentinel-2 Harmonized collection and Cloud Score Plus collection
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');

// Function to preprocess Sentinel-2 with Cloud Score Plus masking and indices calculation
function preprocessS2withIndices(start, end) {
  var filteredS2 = s2.filterBounds(aoi)
                    .filterDate(start, end)
                    .linkCollection(csPlus, [QA_BAND])
                    .map(function(img) {
      // Mask clouds with Cloud Score Plus and scale bands
      var cloudMask = img.select(QA_BAND).gte(CLEAR_THRESHOLD);
      
      // Scale reflectance bands to 0-1 by dividing by 10000
      var scaled = img.select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11', 'B12']).divide(10000);

      return scaled.updateMask(cloudMask).copyProperties(img, img.propertyNames());
    });
  
  var composite = filteredS2.median().clip(aoi).reproject({crs: 'EPSG:32720', scale: 10});
  
  // Calculate spectral indices
  var ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var ndbi = composite.normalizedDifference(['B11', 'B8']).rename('NDBI');
  var ndwi = composite.normalizedDifference(['B3', 'B8']).rename('NDWI');
  var evi = composite.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
    {NIR: composite.select('B8'), RED: composite.select('B4'), BLUE: composite.select('B2')}
  ).rename('EVI');
  var savi = composite.expression(
    '((NIR - RED) / (NIR + RED + 0.5)) * (1.5)',
    {NIR: composite.select('B8'), RED: composite.select('B4')}
  ).rename('SAVI');
  
  return composite.addBands([ndvi, ndbi, ndwi, evi, savi]);
}

// Sentinel-1 preprocessing with indices
function preprocessS1withIndices(start, end) {
  var filteredS1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi)
    .filterDate(start, end)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    .select(['VV', 'VH'])
    .map(function(image) {
      var vvMask = image.select('VV').gt(-30);
      var vhMask = image.select('VH').gt(-30);
      var combinedMask = vvMask.and(vhMask);
      return image.updateMask(combinedMask);
    });

  var composite = filteredS1.median().clip(aoi).reproject({crs: 'EPSG:32720', scale: 10});
//median composite with reprojection and resample
  var ratio = composite.expression('VV / VH', {
    VV: composite.select('VV'),
    VH: composite.select('VH')
  }).rename('VVVH_ratio');

  // Use expression for NDI to avoid masked value issues
  var ndi = composite.expression(
    '(VV - VH) / (VV + VH)', {
      VV: composite.select('VV'),
      VH: composite.select('VH')
    }).rename('NDI');

  return composite.addBands([ratio, ndi]);
}

// Load SRTM DEM and clip it to the study area
var SRTM = ee.Image("USGS/SRTMGL1_003");
var elevation = SRTM.clip(aoi).reproject({crs: 'EPSG:32720', scale: 10});

// Extract slope from SRTM DEM
var slope = ee.Terrain.slope(SRTM).clip(aoi).reproject({crs: 'EPSG:32720', scale: 10});

// Display elevation
Map.addLayer(elevation.select('elevation'), {min:0, max:3000}, 'SRTM Elevation', false);

// Generate feature stacks for 2020 and 2025
var s2_2020 = preprocessS2withIndices(startDate1, endDate1);
var s1_2020 = preprocessS1withIndices(startDate1, endDate1);
var stack2020 = s2_2020.addBands(s1_2020).addBands(elevation).addBands(slope);

print("Feature stack for 2020:", stack2020);

var s2_2025 = preprocessS2withIndices(startDate2, endDate2);
var s1_2025 = preprocessS1withIndices(startDate2, endDate2);
var stack2025 = s2_2025.addBands(s1_2025).addBands(elevation).addBands(slope);

print("Feature stack for 2025:", stack2025);

Map.centerObject(aoi, 11);

var Viz1 = {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.25};
var Viz2 = {bands: ['B11', 'B8', 'B2'], min: 0, max: 0.40};

// Map.addLayer(stack2020.clip(aoi), Viz1, 'Stack 2020 true color');
// Map.addLayer(stack2020.clip(aoi), Viz2, 'Stack 2020 false color');
// Map.addLayer(stack2025.clip(aoi), Viz1, 'Stack 2025 true color');
// Map.addLayer(stack2025.clip(aoi), Viz2, 'Stack 2025 false color');

// print('Resampled projection:', stack2020.projection());
// print('Resampled scale:', stack2020.projection().nominalScale());

var trainingData20 = PForest20.merge(Nonforest20).merge(Water20).merge(SForest20);
var trainingData25 = PForest25.merge(Nonforest25).merge(Water25).merge(SForest25);

print(trainingData20)

var bands20 = stack2020.bandNames();
var bands25 = stack2025.bandNames();

// print("Bands 2020:", bands20);
// print("Bands 2025:", bands25);

/////---Classifying 2020--------------------------------------------/////////
var trainingSamples_2020 = stack2020.select(bands20).sampleRegions({
  collection: trainingData20,
  properties: ['Class20'],
  scale: 10,
  tileScale: 16
});

//print('Training Samples 2020:', trainingSamples_2020)

// Add a column of random uniforms to the training dataset
var withRandom20 = trainingSamples_2020.randomColumn('random', 7);

// Reserve some of the data for model validation, to avoid overfitting the model
var split = 0.7;  // 70% training, 30% testing

var trainingPartition20 = withRandom20.filter(ee.Filter.lt('random', split));
var testingPartition20 = withRandom20.filter(ee.Filter.gte('random', split));

// print('Training samples', trainingPartition20)
// print('Testing samples', testingPartition20)

var classifier20 = ee.Classifier.smileRandomForest(100).train({
  features: trainingPartition20,
  classProperty: 'Class20',
  inputProperties: bands20
});

// Get details of classifier
var explain = classifier20.explain();

// Classify the composite.
var classify2020 = stack2020.classify(classifier20);

// Land cover properties
var value = [1,2,3,4];
//1.Forest, 2.Nonforest, 3.Water, 

var classified2020 = classify2020.rename('LULC20');

var visLULC = {
  min: 1,
  max: 4,
  palette: ['008000', 'ff0000', '0000ff', 'ffc82d']  // green, red, blue for forest, non-forest, water
};

Map.addLayer(classified2020, visLULC, 'LULC 2020', false, 0.7);


// Accuracy Assessment using test samples
var validated20 = testingPartition20.classify(classifier20);

var testAccuracy20 = validated20.errorMatrix('Class20', 'classification');
print('Confusion Matrix 2020:', testAccuracy20);
print('Overall Accuracy 2020:', testAccuracy20.accuracy());

// Compute variable importance
var importance20 = ee.Feature(null, ee.Dictionary(explain).get('importance'));
//print('Variable of Importance', importance20)

// Create a varaiable importance chart
var chart =
  ui.Chart.feature.byProperty(importance20)
  .setChartType('ColumnChart')
  .setOptions({
  title: 'Random Forest Variable Importance 2020',
  legend: {position: 'none'},
  hAxis: {title: 'Bands'},
  vAxis: {title: 'Importance'}
});

// Plot a chart
print("Variable importance in 2020:", chart);

function calcArea(classifiedImage) {
  var areaImage = ee.Image.pixelArea().addBands(classifiedImage);
  var grouped = areaImage.reduceRegion({
    reducer: ee.Reducer.sum().group(1),
    geometry: aoi,
    scale: 10,
    maxPixels: 1e13
  });
  
  var groups = ee.List(grouped.get('groups'));
  return groups.map(function(item) {
    var dict = ee.Dictionary(item);
    var classId = dict.get('group');
    var areaM2 = dict.get('sum');
    var areaHa = ee.Number(areaM2).divide(10000);
    return ee.Feature(null, {'class': classId, 'area_ha': areaHa});
  });
}

var area2020 = calcArea(classified2020);
print('Area by class in 2020 (ha):', area2020);

// Export.image.toAsset({
//   image: classified2020,
//   description: 'Classified2020',
//   region: aoi,
//   scale: 10,
//   crs: 'EPSG:32720',
//   maxPixels: 1e13
// });

/////---Classifying 2025--------------------------------------------/////////
var trainingSamples_2025 = stack2025.select(bands25).sampleRegions({
  collection: trainingData25,
  properties: ['Class25'],
  scale: 10
});

//print('Training Samples 2025:', trainingSamples_2025)

// AAdd a column of random uniforms to the training dataset
var withRandom25 = trainingSamples_2025.randomColumn('random', 7);

// Reserve some of the data for model validation, to avoid overfitting the model
var split = 0.7;  // 70% training, 30% testing

var trainingPartition25 = withRandom25.filter(ee.Filter.lt('random', split));
var testingPartition25 = withRandom25.filter(ee.Filter.gte('random', split));

// print('Training samples', trainingPartition25)
// print('Testing samples', testingPartition25)

var classifier25 = ee.Classifier.smileRandomForest(100).train({
  features: trainingPartition25,
  classProperty: 'Class25',
  inputProperties: bands25
});

// Get details of classifier
var explain25 = classifier25.explain();

// Classify the composite.
var classify2025 = stack2025.classify(classifier25);

// Land cover properties
var value = [1,2,3,4];
//1.Primary Forest, 2.Nonforest, 3.Water, 4. Secondary Forest

var classified2025 = classify2025.rename('LULC25');

var visLULC = {
  min: 1,
  max: 4,
  palette: ['008000', 'ff0000', '0000ff', 'ffc82d']  // green, red, blue for forest, non-forest, water
};

Map.addLayer(classified2025, visLULC, 'LULC 2025', false, 0.7);


// Accuracy Assessment using test samples
var validated25 = testingPartition25.classify(classifier25);

var testAccuracy25 = validated25.errorMatrix('Class25', 'classification');
print('Confusion Matrix 2025:', testAccuracy25);
print('Overall Accuracy 2025:', testAccuracy25.accuracy());

// Compute variable importance
var importance25 = ee.Feature(null, ee.Dictionary(explain25).get('importance'));
//print('Variable of Importance in 2025', importance25)

// Create a varaiable importance chart
var chart =
  ui.Chart.feature.byProperty(importance25)
  .setChartType('ColumnChart')
  .setOptions({
  title: 'Random Forest Variable Importance 2025',
  legend: {position: 'none'},
  hAxis: {title: 'Bands'},
  vAxis: {title: 'Importance'}
});

// Plot a chart
print("Variable importance in 2025:", chart);

function calcArea(classifiedImage) {
  var areaImage = ee.Image.pixelArea().addBands(classifiedImage);
  var grouped = areaImage.reduceRegion({
    reducer: ee.Reducer.sum().group(1),
    geometry: aoi,
    scale: 10,
    maxPixels: 1e13
  });
  
  var groups = ee.List(grouped.get('groups'));
  return groups.map(function(item) {
    var dict = ee.Dictionary(item);
    var classId = dict.get('group');
    var areaM2 = dict.get('sum');
    var areaHa = ee.Number(areaM2).divide(10000);
    return ee.Feature(null, {'class': classId, 'area_ha': areaHa});
  });
}

var area2025 = calcArea(classified2025);
print('Area by class in 2025 (ha):', area2025);

// Export.image.toAsset({
//   image: classified2025,
//   description: 'Classified2025',
//   region: aoi,
//   scale: 10,
//   crs: 'EPSG:32720',
//   maxPixels: 1e13
// });


//---Change Detection----
// Combine classified images into one multi-band image
var changeImage = classified2020.addBands(classified2025).rename(['class_2020', 'class_2025']);

// Create a change matrix: count pixels by (class_2020, class_2025)
var changeMatrix = changeImage.reduceRegion({
  reducer: ee.Reducer.frequencyHistogram().group({
    groupField: 1, // class_2025 band index is 1
    groupName: 'class_2025'
  }),
  geometry: aoi,
  scale: 10,
  maxPixels: 1e13,
}).get('groups');

print('Change Matrix:', changeMatrix);

//Add Legend ----------------------------------------------------------------------------
var panel = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '10px;'
  }
})

var title = ui.Label({
  value: 'Land Cover',
  style: {
    fontSize: '14px',
    fontWeight: 'bold',
    margin: '0px;'
  }
})

panel.add(title);

var color = ['008000', 'ffc82d', 'ff0000', '0000ff'];
var lc_class = [' Primary Forest', 'Secondary Forest', 'Non-forest', 'Water'];

// Cerrado : colour: 'bac2c1', lc_class: 'Cerrado'

var list_legend = function(color, description) {
  
  var c = ui.Label({
    style: {
      backgroundColor: color,
      padding: '10px',
      margin: '4px'
    }
  });
  
  var ds = ui.Label({
    value: description,
    style: {
      margin: '5px'
    }
  });
  
  return ui.Panel({
    widgets: [c, ds],
    layout: ui.Panel.Layout.Flow('horizontal')
  });
};

for(var a = 0; a < 4; a++){
  panel.add(list_legend(color[a], lc_class[a]));
}

Map.add(panel);
