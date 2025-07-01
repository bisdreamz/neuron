package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.Optimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * DEBUG version of CorrectProductionScenarioTest using raw integer features.
 * 
 * This test bypasses all feature processing (embeddings, one-hot encoding) by using
 * raw integers as passthrough features. This helps isolate whether the learning
 * failure is due to feature processing or the core neural network.
 * 
 * Key differences:
 * - All features are passthrough (no embeddings or one-hot)
 * - Input values are raw integers cast to floats
 * - Network learns directly from numeric feature values
 */
public class DebugCorrectProductionScenarioTest {
    
    // Segment configuration
    private static final int numPremiumSegments = 100;  // 100 premium segments
    private static final int numRegularSegments = 5000; // 5000 regular segments
    private final Set<String> premiumSegments = new HashSet<>();
    private final Set<String> regularSegments = new HashSet<>();
    
    // Tracking maps
    private final Map<String, Integer> segmentPenaltyCounts = new HashMap<>();
    private final Map<String, Integer> segmentBidCounts = new HashMap<>();
    private final Map<String, Float> segmentTotalBids = new HashMap<>();
    private final Map<String, Float> segmentCPM = new HashMap<>();
    
    // New tracking for true averages
    private final Map<String, Integer> segmentTotalTrainingCounts = new HashMap<>();
    private final Map<String, Float> segmentTotalTrainingValues = new HashMap<>();
    
    @Test
    public void testDebugWithRawIntegers() {
        System.out.println("=== DEBUG: RAW INTEGER FEATURES TEST ===\n");
        System.out.println("What actually happens:");
        System.out.println("1. EVERY request → penalty training");
        System.out.println("2. ~50% of requests → ALSO bid training");
        System.out.println("3. Premium segments get MORE traffic");
        System.out.println("4. Overall ratio: 2:1 penalties:bids\n");
        
        // Test with different penalty approaches and batch sizes
        int batchSize = 1; // Single sample training like production
        testWithPenalty("Negative penalty (-$0.0003)", -0.0003f, batchSize);
        //testWithPenalty("Tiny positive ($0.0001)", 0.0001f, batchSize);
        //testWithPenalty("Very tiny positive ($0.00001)", 0.00001f, batchSize);
    }
    
    private void testWithPenalty(String name, float penaltyValue, int batchSize) {
        System.out.printf("\n=== %s ===\n", name);
        
        // Clear tracking maps for new test
        segmentPenaltyCounts.clear();
        segmentBidCounts.clear();
        segmentTotalBids.clear();
        segmentCPM.clear();
        premiumSegments.clear();
        regularSegments.clear();
        segmentTotalTrainingCounts.clear();
        segmentTotalTrainingValues.clear();
        
        // Initialize segments with CPM values
        Random initRand = new Random(42);
        initializeSegmentCPMs(initRand);
        
        System.out.printf("Batch size: %d, Penalty value: $%.5f\n", batchSize, penaltyValue);
        System.out.println("Using raw integer passthrough features (no embeddings or one-hot)");

        Feature[] features = {
            Feature.autoNormalize( "OS"),      // Raw integer 0-3
            Feature.autoNormalize("ZONEID"),  // Raw integer 0-9999
            Feature.autoNormalize("DOMAIN"),  // Raw integer 0-4999
            Feature.autoNormalize("PUB")      // Raw integer 0-1999
        };

        //Optimizer optimizer = new SgdOptimizer(0.001f); // Same as updated SerialCorrectProductionScenarioTest
        Optimizer optimizer = new AdamWOptimizer(0.001f, 0.0001f);

        NeuralNet net = NeuralNet.newBuilder()
                .setDefaultOptimizer(optimizer)
                .layer(Layers.inputAllNumerical(features.length, Arrays.stream(features).map(f -> f.getName()).toArray(String[]::new)))
                .layer(Layers.hiddenDenseRelu(512))
                .layer(Layers.hiddenDenseLeakyRelu(256))
                .layer(Layers.hiddenDenseLeakyRelu(128))
                .layer(Layers.hiddenDenseLeakyRelu(64))
                .withGlobalGradientClipping(0f)
                .output(Layers.outputLinearRegression(1));

        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);

        Random rand = new Random(42);

        // Track actual CPM values by feature combination
        Map<String, List<Float>> featureComboCPMs = new HashMap<>();
        Map<Object, List<Float>> cpmByOS = new HashMap<>();

        // Tracking
        int totalPenaltyTrains = 0;
        int totalBidTrains = 0;
        
        // Histogram tracking for CPM values
        Map<String, Integer> cpmHistogram = new TreeMap<>();
        
        // Batch accumulation
        List<Map<String, Object>> batchInputs = new ArrayList<>();
        List<Float> batchTargets = new ArrayList<>();

        int steps = 20000;  // Match SerialCorrectProductionScenarioTest
        System.out.println("Simulating " + steps + " requests (with smaller one-hot vocab)...");

        for (int step = 0; step < steps; step++) {
            // Weighted selection: 30% chance for premium segments, 70% for regular
            String segment;
            int zoneId, domainId;
            boolean isPremium = rand.nextFloat() < 0.3f;

            if (isPremium) {
                // Pick a random premium segment
                int premiumIndex = rand.nextInt(numPremiumSegments);
                zoneId = premiumIndex;
                domainId = premiumIndex % 1000;
            } else {
                // Pick a random regular segment
                zoneId = numPremiumSegments + rand.nextInt(numRegularSegments);
                domainId = zoneId % 5000;
            }
            segment = zoneId + "_" + domainId;

            int os = rand.nextInt(4);
            int pubId = rand.nextInt(20);  // Only 20 publishers
            Map<String, Object> input = Map.of(
                "OS", os,
                "ZONEID", zoneId,
                "DOMAIN", domainId,
                "PUB", pubId
            );
            
            // STEP 1: ALWAYS train with penalty (auction cost)
            batchInputs.add(input);
            batchTargets.add(penaltyValue);
            totalPenaltyTrains++;
            segmentPenaltyCounts.merge(segment, 1, Integer::sum);
            
            // Track for true average calculation
            segmentTotalTrainingCounts.merge(segment, 1, Integer::sum);
            segmentTotalTrainingValues.merge(segment, penaltyValue, Float::sum);
            
            // STEP 2: ~50% of requests also get bid training
            if (rand.nextFloat() < 0.5f) {
                float bidRate = isPremium ? 0.8f : 0.05f;
                float bidValue;
                
                if (rand.nextFloat() < bidRate) {
                    // Win bid - use pre-assigned CPM with some noise
                    float baseCPM = segmentCPM.get(segment);
                    float noise = (rand.nextFloat() - 0.5f) * 0.4f; // +/- 20% variation
                    bidValue = Math.max(0.0f, baseCPM + noise);
                } else {
                    // No bid
                    bidValue = 0.0f;
                }
                
                // Add bid training to batch
                batchInputs.add(new HashMap<>(input)); // Clone to avoid overwriting
                batchTargets.add(bidValue);
                totalBidTrains++;
                segmentBidCounts.merge(segment, 1, Integer::sum);
                segmentTotalBids.merge(segment, bidValue, Float::sum);
                
                // Track for true average calculation
                segmentTotalTrainingCounts.merge(segment, 1, Integer::sum);
                segmentTotalTrainingValues.merge(segment, bidValue, Float::sum);
                
                // Track CPM distribution
                String bucket = String.format("$%.1f-%.1f", Math.floor(bidValue), Math.floor(bidValue) + 1);
                cpmHistogram.merge(bucket, 1, Integer::sum);
                
                // Track CPM by OS
                cpmByOS.computeIfAbsent(os, k -> new ArrayList<>()).add(bidValue);
            }
            
            // Train batch when full or at end of data
            if (batchInputs.size() >= batchSize || step == steps - 1) {
                if (!batchInputs.isEmpty()) {
                    // Train the batch using SimpleNetFloat's batch training
                    model.trainBatchMaps(batchInputs, batchTargets);
                    
                    // Clear batch
                    batchInputs.clear();
                    batchTargets.clear();
                }
            }
            
            // Progress check
            if (step > 0 && step % 100 == 0) {  // More frequent updates
                System.out.printf("Step %d: %d penalties, %d bids (ratio %.1f:1)\n",
                    step, totalPenaltyTrains, totalBidTrains,
                    (float)totalPenaltyTrains / totalBidTrains);
                
                // Test some predictions with raw integers
                float premiumPred = model.predictFloat(Map.of(
                        "OS", 0, "ZONEID", 10, "DOMAIN", 10, "PUB", 10));
                float regularPred = model.predictFloat(Map.of(
                        "OS", 0, "ZONEID", 5000, "DOMAIN", 0, "PUB", 1500));
                
                System.out.printf("  Premium: $%.3f, Regular: $%.3f\n", premiumPred, regularPred);
            }
        }
        
        // Final statistics
        System.out.printf("\nFinal: %d penalty trains, %d bid trains (ratio %.1f:1)\n",
            totalPenaltyTrains, totalBidTrains,
            (float)totalPenaltyTrains / totalBidTrains);
        
        // Print CPM histogram
        System.out.println("\nCPM Distribution Histogram:");
        for (Map.Entry<String, Integer> entry : cpmHistogram.entrySet()) {
            int count = entry.getValue();
            System.out.printf("%s: %4d [", entry.getKey(), count);
            for (int i = 0; i < Math.min(50, count / 10); i++) System.out.print("█");
            System.out.println("]");
        }
        
        // Analyze traffic distribution
        int premiumTraffic = 0;
        int regularTraffic = 0;
        for (Map.Entry<String, Integer> entry : segmentPenaltyCounts.entrySet()) {
            if (premiumSegments.contains(entry.getKey())) {
                premiumTraffic += entry.getValue();
            } else {
                regularTraffic += entry.getValue();
            }
        }
        
        System.out.printf("Traffic distribution: Premium %.1f%%, Regular %.1f%%\n",
            100.0 * premiumTraffic / totalPenaltyTrains,
            100.0 * regularTraffic / totalPenaltyTrains);
        
        // Show CPM distribution by OS
        System.out.println("\nCPM Distribution by OS:");
        for (int i = 0; i < 4; i++) {
            List<Float> cpms = cpmByOS.get(i);
            if (cpms != null && !cpms.isEmpty()) {
                float avgCPM = cpms.stream().reduce(0f, Float::sum) / cpms.size();
                float maxCPM = cpms.stream().max(Float::compare).orElse(0f);
                System.out.printf("  OS %d: avg=$%.2f, max=$%.2f, count=%d\n", 
                    i, avgCPM, maxCPM, cpms.size());
            }
        }
        
        // Show penalty vs CPM contribution analysis
        System.out.println("\nValue Contribution Analysis:");
        analyzeValueContributions(penaltyValue);
        
        // Print sample segment CPMs to verify data generation
        System.out.println("\nSample segment CPMs (first 10 premium, 10 regular):");
        int count = 0;
        for (Map.Entry<String, Float> entry : segmentCPM.entrySet()) {
            if (count < 10 && premiumSegments.contains(entry.getKey())) {
                System.out.printf("  Premium %s: $%.2f\n", entry.getKey(), entry.getValue());
                count++;
            }
        }
        count = 0;
        for (Map.Entry<String, Float> entry : segmentCPM.entrySet()) {
            if (count < 10 && regularSegments.contains(entry.getKey())) {
                System.out.printf("  Regular %s: $%.2f\n", entry.getKey(), entry.getValue());
                count++;
            }
        }
        
        // Test final predictions
        System.out.println("\nTesting prediction accuracy:");
        evaluatePredictions(model, segmentCPM, segmentBidCounts, segmentTotalBids, penaltyValue);
        
        // Show detailed histogram analysis
        evaluatePredictionsWithHistogram(model, penaltyValue);
    }
    
    private void evaluatePredictions(SimpleNetFloat model, Map<String, Float> segmentCPM,
                                    Map<String, Integer> segmentBidCounts,
                                    Map<String, Float> segmentTotalBids, float penaltyValue) {
        List<Float> premiumPreds = new ArrayList<>();
        List<Float> premiumExpected = new ArrayList<>();
        List<Float> regularPreds = new ArrayList<>();
        List<Float> regularExpected = new ArrayList<>();
        
        // Test premium segments
        for (int i = 0; i < 50; i++) {
            String segment = i + "_" + i;
            float pred = model.predictFloat(Map.of(
                "OS", 0, "ZONEID", i, "DOMAIN", i, "PUB", i));
            premiumPreds.add(pred);
            
            // Calculate expected based on all training values (penalty + CPM)
            if (segmentTotalTrainingCounts.containsKey(segment)) {
                float avgTrainingValue = segmentTotalTrainingValues.get(segment) / segmentTotalTrainingCounts.get(segment);
                premiumExpected.add(avgTrainingValue);
            }
        }
        
        // Test regular segments
        for (int i = 5000; i < 5050; i++) {
            int domainId = i % 5000;
            String segment = i + "_" + domainId;
            float pred = model.predictFloat(Map.of(
                "OS", 0, "ZONEID", i, "DOMAIN", domainId, "PUB", 1000 + (i % 1000)));
            regularPreds.add(pred);
            
            if (segmentTotalTrainingCounts.containsKey(segment)) {
                float avgTrainingValue = segmentTotalTrainingValues.get(segment) / segmentTotalTrainingCounts.get(segment);
                regularExpected.add(avgTrainingValue);
            }
        }
        
        // Calculate averages
        float premiumAvgPred = premiumPreds.stream().reduce(0f, Float::sum) / premiumPreds.size();
        float regularAvgPred = regularPreds.stream().reduce(0f, Float::sum) / regularPreds.size();
        
        // Calculate expected averages from actual training data
        float premiumAvgExpected = 0;
        float regularAvgExpected = 0;
        
        if (!premiumExpected.isEmpty()) {
            premiumAvgExpected = premiumExpected.stream().reduce(0f, Float::sum) / premiumExpected.size();
        } else {
            // Estimate based on typical premium patterns
            float typicalPremiumCPM = 3.5f;
            float premiumBidRate = 0.8f;
            premiumAvgExpected = penaltyValue + (premiumBidRate * typicalPremiumCPM);
        }
        
        if (!regularExpected.isEmpty()) {
            regularAvgExpected = regularExpected.stream().reduce(0f, Float::sum) / regularExpected.size();
        } else {
            // Estimate based on typical regular patterns
            float typicalRegularCPM = 0.3f;
            float regularBidRate = 0.05f;
            regularAvgExpected = penaltyValue + (regularBidRate * typicalRegularCPM);
        }
        
        // Check uniqueness
        Set<String> uniquePremium = new HashSet<>();
        Set<String> uniqueRegular = new HashSet<>();
        for (float p : premiumPreds) uniquePremium.add(String.format("%.3f", p));
        for (float p : regularPreds) uniqueRegular.add(String.format("%.3f", p));
        
        System.out.printf("Premium: predicted avg=$%.3f (expected ~$%.3f), %d unique values\n",
            premiumAvgPred, premiumAvgExpected, uniquePremium.size());
        System.out.printf("Regular: predicted avg=$%.3f (expected ~$%.3f), %d unique values\n",
            regularAvgPred, regularAvgExpected, uniqueRegular.size());
        System.out.printf("Differentiation: $%.3f\n", premiumAvgPred - regularAvgPred);
        
        // Success criteria
        boolean success = premiumAvgPred > regularAvgPred + 0.5f && // Clear differentiation
                         premiumAvgPred > 1.0f && // Premium predicts high
                         uniquePremium.size() > 10; // Good diversity
        
        System.out.println(success ? "[SUCCESS] - Model differentiates segments!" : "[FAILURE] - Model cannot differentiate!");
    }
    
    private record PredictionResult(float premiumPred, float regularPred, int step) {}
    
    private void analyzeValueContributions(float penaltyValue) {
        // Analyze a few sample segments
        int premiumSamples = 0;
        float totalPremiumExpected = 0;
        float totalPremiumCPMContribution = 0;
        
        for (String segment : premiumSegments) {
            if (segmentTotalTrainingCounts.containsKey(segment) && premiumSamples < 20) {
                float avgTrainingValue = segmentTotalTrainingValues.get(segment) / segmentTotalTrainingCounts.get(segment);
                int penaltyCount = segmentPenaltyCounts.getOrDefault(segment, 0);
                int bidCount = segmentBidCounts.getOrDefault(segment, 0);
                float avgBidValue = bidCount > 0 ? segmentTotalBids.get(segment) / bidCount : 0;
                
                // Calculate contributions
                float penaltyContribution = penaltyValue * penaltyCount / (float)(penaltyCount + bidCount);
                float cpmContribution = avgBidValue * bidCount / (float)(penaltyCount + bidCount);
                
                totalPremiumExpected += avgTrainingValue;
                totalPremiumCPMContribution += cpmContribution;
                premiumSamples++;
            }
        }
        
        int regularSamples = 0;
        float totalRegularExpected = 0;
        float totalRegularCPMContribution = 0;
        
        for (String segment : regularSegments) {
            if (segmentTotalTrainingCounts.containsKey(segment) && regularSamples < 20) {
                float avgTrainingValue = segmentTotalTrainingValues.get(segment) / segmentTotalTrainingCounts.get(segment);
                int penaltyCount = segmentPenaltyCounts.getOrDefault(segment, 0);
                int bidCount = segmentBidCounts.getOrDefault(segment, 0);
                float avgBidValue = bidCount > 0 ? segmentTotalBids.get(segment) / bidCount : 0;
                
                // Calculate contributions
                float penaltyContribution = penaltyValue * penaltyCount / (float)(penaltyCount + bidCount);
                float cpmContribution = avgBidValue * bidCount / (float)(penaltyCount + bidCount);
                
                totalRegularExpected += avgTrainingValue;
                totalRegularCPMContribution += cpmContribution;
                regularSamples++;
            }
        }
        
        if (premiumSamples > 0) {
            float avgPremiumExpected = totalPremiumExpected / premiumSamples;
            float avgPremiumCPM = totalPremiumCPMContribution / premiumSamples;
            System.out.printf("  Premium segments: Expected avg=$%.3f (penalty=$%.3f + CPM contribution=$%.3f)\n",
                avgPremiumExpected, penaltyValue, avgPremiumCPM);
        }
        
        if (regularSamples > 0) {
            float avgRegularExpected = totalRegularExpected / regularSamples;
            float avgRegularCPM = totalRegularCPMContribution / regularSamples;
            System.out.printf("  Regular segments: Expected avg=$%.3f (penalty=$%.3f + CPM contribution=$%.3f)\n",
                avgRegularExpected, penaltyValue, avgRegularCPM);
        }
    }
    
    private void initializeSegmentCPMs(Random rand) {
        // Initialize premium segments with high CPMs distributed across features
        for (int i = 0; i < numPremiumSegments; i++) {
            String segment = i + "_" + i;
            premiumSegments.add(segment);
            
            // Base CPM $2-5, but vary by index to ensure distribution
            float baseCPM = 2.0f + (i % 4) * 0.75f; // Distributes across 2.0, 2.75, 3.5, 4.25
            float randomVariation = rand.nextFloat() * 1.0f; // Add 0-1.0 random
            segmentCPM.put(segment, baseCPM + randomVariation);
        }
        
        // Initialize regular segments with low CPMs
        for (int i = 0; i < numRegularSegments; i++) {
            int zoneId = numPremiumSegments + i;
            int domainId = zoneId % 5000;
            String segment = zoneId + "_" + domainId;
            regularSegments.add(segment);
            
            // Base CPM $0.1-0.5
            float baseCPM = 0.1f + rand.nextFloat() * 0.4f;
            segmentCPM.put(segment, baseCPM);
        }
        
        System.out.printf("Initialized %d premium segments (CPM $2-5) and %d regular segments (CPM $0.1-0.5)\n",
            premiumSegments.size(), regularSegments.size());
    }
    
    private void evaluatePredictionsWithHistogram(SimpleNetFloat model, float penaltyValue) {
        System.out.println("\n=== ACTUAL vs PREDICTED VALUE ANALYSIS ===");
        
        List<Float> allActuals = new ArrayList<>();
        List<Float> allPredictions = new ArrayList<>();
        List<Float> percentageErrors = new ArrayList<>();
        
        // Test a sample of segments
        Random testRand = new Random(123);
        int samplesPerType = 100;
        
        // Test premium segments
        List<String> premiumList = new ArrayList<>(premiumSegments);
        Collections.shuffle(premiumList, testRand);
        for (int i = 0; i < Math.min(samplesPerType, premiumList.size()); i++) {
            String segment = premiumList.get(i);
            String[] parts = segment.split("_");
            int zoneId = Integer.parseInt(parts[0]);
            int domainId = Integer.parseInt(parts[1]);
            
            // Test with different OS values to ensure distribution
            for (int os = 0; os < 4; os++) {
                Map<String, Object> input = Map.of(
                    "OS", os,
                    "ZONEID", zoneId,
                    "DOMAIN", domainId,
                    "PUB", zoneId % 2000
                );
                
                // Get the true expected value (average of all training values)
                float actual;
                if (segmentTotalTrainingCounts.containsKey(segment)) {
                    actual = segmentTotalTrainingValues.get(segment) / segmentTotalTrainingCounts.get(segment);
                } else {
                    // If not trained, use CPM + penalty estimate
                    actual = penaltyValue + (0.8f * segmentCPM.get(segment));
                }
                
                float predicted = model.predictFloat(input);
                
                allActuals.add(actual);
                allPredictions.add(predicted);
                
                // Calculate percentage error
                float error = actual != 0 ? (predicted - actual) / Math.abs(actual) * 100 : 0;
                percentageErrors.add(error);
            }
        }
        
        // Test regular segments
        List<String> regularList = new ArrayList<>(regularSegments);
        Collections.shuffle(regularList, testRand);
        for (int i = 0; i < Math.min(samplesPerType, regularList.size()); i++) {
            String segment = regularList.get(i);
            String[] parts = segment.split("_");
            int zoneId = Integer.parseInt(parts[0]);
            int domainId = Integer.parseInt(parts[1]);
            
            Map<String, Object> input = Map.of(
                "OS", testRand.nextInt(4),
                "ZONEID", zoneId,
                "DOMAIN", domainId,
                "PUB", zoneId % 2000
            );
            
            // Get the true expected value (average of all training values)
            float actual;
            if (segmentTotalTrainingCounts.containsKey(segment)) {
                actual = segmentTotalTrainingValues.get(segment) / segmentTotalTrainingCounts.get(segment);
            } else {
                // If not trained, use CPM + penalty estimate
                actual = penaltyValue + (0.05f * segmentCPM.get(segment));
            }
            
            float predicted = model.predictFloat(input);
            
            allActuals.add(actual);
            allPredictions.add(predicted);
            
            float error = actual != 0 ? (predicted - actual) / Math.abs(actual) * 100 : 0;
            percentageErrors.add(error);
        }
        
        // Create histogram buckets for actual vs predicted
        Map<String, Integer> actualHistogram = new TreeMap<>();
        Map<String, Integer> predictedHistogram = new TreeMap<>();
        
        for (int i = 0; i < allActuals.size(); i++) {
            String actualBucket = String.format("$%.1f-%.1f", 
                Math.floor(allActuals.get(i)), Math.floor(allActuals.get(i)) + 1);
            String predictedBucket = String.format("$%.1f-%.1f", 
                Math.floor(allPredictions.get(i)), Math.floor(allPredictions.get(i)) + 1);
            
            actualHistogram.merge(actualBucket, 1, Integer::sum);
            predictedHistogram.merge(predictedBucket, 1, Integer::sum);
        }
        
        // Print side-by-side histogram
        System.out.println("\nActual vs Predicted Value Distribution:");
        System.out.println("Value Range | Actual Count | Predicted Count");
        System.out.println("----------|--------------|----------------");
        
        Set<String> allBuckets = new TreeSet<>();
        allBuckets.addAll(actualHistogram.keySet());
        allBuckets.addAll(predictedHistogram.keySet());
        
        for (String bucket : allBuckets) {
            int actualCount = actualHistogram.getOrDefault(bucket, 0);
            int predictedCount = predictedHistogram.getOrDefault(bucket, 0);
            System.out.printf("%-9s | %12d | %15d\n", bucket, actualCount, predictedCount);
        }
        
        // Percentage error histogram
        Map<String, Integer> errorHistogram = new TreeMap<>();
        for (float error : percentageErrors) {
            String bucket;
            if (error < -50) bucket = "< -50%";
            else if (error < -25) bucket = "-50% to -25%";
            else if (error < -10) bucket = "-25% to -10%";
            else if (error < 0) bucket = "-10% to 0%";
            else if (error < 10) bucket = "0% to +10%";
            else if (error < 25) bucket = "+10% to +25%";
            else if (error < 50) bucket = "+25% to +50%";
            else bucket = "> +50%";
            
            errorHistogram.merge(bucket, 1, Integer::sum);
        }
        
        System.out.println("\nPrediction Error Distribution:");
        System.out.println("Error Range    | Count | Visual");
        System.out.println("---------------|-------|--------");
        for (Map.Entry<String, Integer> entry : errorHistogram.entrySet()) {
            int count = entry.getValue();
            System.out.printf("%-14s | %5d | ", entry.getKey(), count);
            for (int i = 0; i < Math.min(50, count / 10); i++) System.out.print("█");
            System.out.println();
        }
        
        // Calculate statistics
        float meanAbsoluteError = 0;
        float meanAbsolutePercentageError = 0;
        float rmse = 0;
        
        for (int i = 0; i < allActuals.size(); i++) {
            float error = allPredictions.get(i) - allActuals.get(i);
            meanAbsoluteError += Math.abs(error);
            meanAbsolutePercentageError += Math.abs(percentageErrors.get(i));
            rmse += error * error;
        }
        
        meanAbsoluteError /= allActuals.size();
        meanAbsolutePercentageError /= allActuals.size();
        rmse = (float) Math.sqrt(rmse / allActuals.size());
        
        System.out.printf("\nError Statistics:\n");
        System.out.printf("  Mean Absolute Error (MAE): $%.4f\n", meanAbsoluteError);
        System.out.printf("  Mean Absolute Percentage Error (MAPE): %.2f%%\n", meanAbsolutePercentageError);
        System.out.printf("  Root Mean Square Error (RMSE): $%.4f\n", rmse);
        
        // Check if model is learning properly
        float avgPremiumActual = 0, avgPremiumPred = 0;
        float avgRegularActual = 0, avgRegularPred = 0;
        int premiumCount = 0, regularCount = 0;
        
        for (int i = 0; i < allActuals.size(); i++) {
            if (allActuals.get(i) > 1.5f) { // Premium
                avgPremiumActual += allActuals.get(i);
                avgPremiumPred += allPredictions.get(i);
                premiumCount++;
            } else { // Regular
                avgRegularActual += allActuals.get(i);
                avgRegularPred += allPredictions.get(i);
                regularCount++;
            }
        }
        
        avgPremiumActual /= premiumCount;
        avgPremiumPred /= premiumCount;
        avgRegularActual /= regularCount;
        avgRegularPred /= regularCount;
        
        System.out.printf("\nSegment Analysis:\n");
        System.out.printf("  Premium - Actual: $%.3f, Predicted: $%.3f (%.1f%% error)\n", 
            avgPremiumActual, avgPremiumPred, 
            (avgPremiumPred - avgPremiumActual) / avgPremiumActual * 100);
        System.out.printf("  Regular - Actual: $%.3f, Predicted: $%.3f (%.1f%% error)\n", 
            avgRegularActual, avgRegularPred,
            (avgRegularPred - avgRegularActual) / avgRegularActual * 100);
        
        boolean success = meanAbsolutePercentageError < 25.0f && 
                         avgPremiumPred > avgRegularPred * 2;
        
        System.out.println(success ? "\n[SUCCESS] Model learns CPM patterns accurately!" : 
                                    "\n[FAILURE] Model struggles to learn CPM patterns!");
    }
}