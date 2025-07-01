package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test for mode collapse with production-like complexity.
 * Tests both online and batch training with mixed feature types and cardinalities.
 */
public class ComprehensiveModeCollapseTest {
    
    @Test
    public void testOnlineTrainingMixedFeatures() {
        System.out.println("=== COMPREHENSIVE ONLINE TRAINING TEST ===\n");
        
        // Production-like feature configuration with 7 features
        Feature[] features = {
            // High cardinality embeddings
            Feature.hashedEmbedding(50_000, 32, "app_bundle"),      // Very high cardinality
            Feature.hashedEmbedding(10_000, 16, "domain"),         // High cardinality
            Feature.embeddingLRU(4000, 16, "zone_id"),             // Medium-high cardinality
            
            // Low cardinality one-hot
            Feature.oneHot(25, "country"),                          // ~20 countries
            Feature.oneHot(5, "device_type"),                       // phone/tablet/tv/desktop/other
            Feature.oneHot(4, "os"),                                // ios/android/roku/other
            
            // Continuous features
            Feature.passthrough("bid_floor"),                       // 0.01 - 0.5
            Feature.autoScale(0f, 24f, "hour_of_day")              // 0-23
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.dropout(0.2f))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Simulate production data distributions
        ProductionSimulator sim = new ProductionSimulator();
        
        System.out.println("Training with online approach (1 sample at a time)...");
        int totalSamples = 100_000;
        int bidCount = 0;
        int noBidCount = 0;
        
        for (int i = 0; i < totalSamples; i++) {
            Map<String, Object> request = sim.generateRequest();
            
            // Production logic: 1.5% bid rate
            if (sim.shouldBid(request)) {
                float bidValue = sim.getBidValue(request);
                model.train(request, bidValue);
                bidCount++;
            } else if (sim.rand.nextFloat() < 0.02f) { // 2% of no-bids get negative training
                model.train(request, -0.25f);
                noBidCount++;
            }
            
            // Check for mode collapse every 10k samples
            if (i > 0 && i % 10_000 == 0) {
                System.out.printf("\nProgress: %d samples, %d bids, %d no-bids\n", i, bidCount, noBidCount);
                PredictionAnalysis analysis = analyzePredictions(model, sim, 1000);
                analysis.print();
                
                if (analysis.hasCollapsed()) {
                    fail("Mode collapse detected at sample " + i + ": " + analysis.getCollapseReason());
                }
            }
        }
        
        System.out.printf("\nFinal: %d total, %d bids (%.1f%%), %d no-bids\n", 
            totalSamples, bidCount, 100.0 * bidCount / totalSamples, noBidCount);
        
        // Final comprehensive evaluation
        System.out.println("\n=== FINAL EVALUATION ===");
        evaluateModel(model, sim);
    }
    
    @Test
    public void testBatchTrainingMixedFeatures() {
        System.out.println("=== COMPREHENSIVE BATCH TRAINING TEST ===\n");
        
        // Same feature configuration
        Feature[] features = {
            Feature.hashedEmbedding(50_000, 32, "app_bundle"),
            Feature.hashedEmbedding(10_000, 16, "domain"),
            Feature.embeddingLRU(4000, 16, "zone_id"),
            Feature.oneHot(25, "country"),
            Feature.oneHot(5, "device_type"),
            Feature.oneHot(4, "os"),
            Feature.passthrough("bid_floor"),
            Feature.autoScale(0f, 24f, "hour_of_day")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.dropout(0.2f))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        ProductionSimulator sim = new ProductionSimulator();
        
        // Collect training data first
        System.out.println("Collecting training data...");
        List<TrainingExample> trainingData = new ArrayList<>();
        
        for (int i = 0; i < 100_000; i++) {
            Map<String, Object> request = sim.generateRequest();
            
            if (sim.shouldBid(request)) {
                float bidValue = sim.getBidValue(request);
                trainingData.add(new TrainingExample(request, bidValue));
            } else if (sim.rand.nextFloat() < 0.02f) {
                trainingData.add(new TrainingExample(request, -0.25f));
            }
        }
        
        System.out.printf("Collected %d training examples\n", trainingData.size());
        
        // Train in batches
        System.out.println("\nTraining in batches...");
        Collections.shuffle(trainingData, sim.rand);
        
        int epochs = 3;
        int batchSize = 32;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.printf("\nEpoch %d/%d\n", epoch + 1, epochs);
            
            for (int i = 0; i < trainingData.size(); i += batchSize) {
                int end = Math.min(i + batchSize, trainingData.size());
                
                // Train each example in the batch
                for (int j = i; j < end; j++) {
                    TrainingExample ex = trainingData.get(j);
                    model.train(ex.features, ex.target);
                }
                
                // Progress check
                if (i % 1000 == 0) {
                    System.out.printf("  Batch %d/%d\n", i / batchSize, trainingData.size() / batchSize);
                }
            }
            
            // Check after each epoch
            PredictionAnalysis analysis = analyzePredictions(model, sim, 1000);
            System.out.println("\nEnd of epoch " + (epoch + 1) + ":");
            analysis.print();
            
            if (analysis.hasCollapsed()) {
                fail("Mode collapse detected at epoch " + (epoch + 1) + ": " + analysis.getCollapseReason());
            }
        }
        
        // Final evaluation
        System.out.println("\n=== FINAL EVALUATION ===");
        evaluateModel(model, sim);
    }
    
    @Test
    public void testHighNegativeRatioTraining() {
        System.out.println("=== HIGH NEGATIVE RATIO TEST (Mimics Production) ===\n");
        
        Feature[] features = {
            Feature.hashedEmbedding(50_000, 32, "app_bundle"),
            Feature.embeddingLRU(4000, 16, "zone_id"),
            Feature.oneHot(25, "country"),
            Feature.oneHot(4, "os"),
            Feature.passthrough("bid_floor")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        ProductionSimulator sim = new ProductionSimulator();
        
        // Train with exact production ratios
        System.out.println("Training with production ratios:");
        System.out.println("- 1% of requests result in bids (0.5-3.0)");
        System.out.println("- 2% of non-bids get negative training (-0.01)");
        System.out.println("- Effective ratio: ~66% negative, ~33% positive\n");
        
        int totalRequests = 50_000;
        int bidCount = 0;
        int negativeCount = 0;
        
        for (int i = 0; i < totalRequests; i++) {
            Map<String, Object> request = sim.generateRequest(false);
            
            // Exactly 1% bid rate
            if (sim.rand.nextFloat() < 0.01f) {
                float bidValue = 0.5f + sim.rand.nextFloat() * 2.5f;
                model.train(request, bidValue);
                bidCount++;
            } else if (sim.rand.nextFloat() < 0.02f) {
                // Use -0.01 like production
                model.train(request, -0.01f);
                negativeCount++;
            }
            
            if (i > 0 && i % 5000 == 0) {
                System.out.printf("Samples: %d, Bids: %d, Negatives: %d\n", i, bidCount, negativeCount);
                
                // Sample predictions
                Set<String> uniquePreds = new HashSet<>();
                float sum = 0;
                for (int j = 0; j < 100; j++) {
                    float pred = model.predictFloat(sim.generateRequest(false));
                    uniquePreds.add(String.format("%.3f", pred));
                    sum += pred;
                }
                
                System.out.printf("  Unique predictions: %d/100, Average: %.3f\n", 
                    uniquePreds.size(), sum / 100);
                
                if (uniquePreds.size() < 5) {
                    System.out.println("  WARNING: Low prediction diversity!");
                }
            }
        }
        
        // Test with different negative values
        System.out.println("\n=== Testing Different Negative Values ===");
        testNegativeValue(features, optimizer, -0.01f, "Very weak negative (-0.01)");
        testNegativeValue(features, optimizer, -0.1f, "Weak negative (-0.1)");
        testNegativeValue(features, optimizer, -0.25f, "Moderate negative (-0.25)");
        testNegativeValue(features, optimizer, -0.5f, "Strong negative (-0.5)");
    }
    
    private void testNegativeValue(Feature[] features, AdamWOptimizer optimizer, 
                                  float negativeValue, String description) {
        System.out.println("\n" + description + ":");
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        ProductionSimulator sim = new ProductionSimulator();
        
        // Quick training
        for (int i = 0; i < 10_000; i++) {
            Map<String, Object> request = sim.generateRequest(false);
            
            if (sim.rand.nextFloat() < 0.01f) {
                model.train(request, 0.5f + sim.rand.nextFloat() * 2.5f);
            } else if (sim.rand.nextFloat() < 0.02f) {
                model.train(request, negativeValue);
            }
        }
        
        // Check diversity
        Set<String> uniquePreds = new HashSet<>();
        float min = Float.MAX_VALUE, max = Float.MIN_VALUE;
        for (int i = 0; i < 200; i++) {
            float pred = model.predictFloat(sim.generateRequest(false));
            uniquePreds.add(String.format("%.3f", pred));
            min = Math.min(min, pred);
            max = Math.max(max, pred);
        }
        
        System.out.printf("  Unique: %d/200, Range: [%.3f, %.3f]\n", 
            uniquePreds.size(), min, max);
    }
    
    private PredictionAnalysis analyzePredictions(SimpleNetFloat model, ProductionSimulator sim, int samples) {
        Map<String, Integer> predictionCounts = new HashMap<>();
        float sum = 0, sumSquares = 0;
        float min = Float.MAX_VALUE, max = Float.MIN_VALUE;
        
        for (int i = 0; i < samples; i++) {
            Map<String, Object> request = sim.generateRequest();
            float pred = model.predictFloat(request);
            
            String rounded = String.format("%.3f", pred);
            predictionCounts.merge(rounded, 1, Integer::sum);
            
            sum += pred;
            sumSquares += pred * pred;
            min = Math.min(min, pred);
            max = Math.max(max, pred);
        }
        
        float mean = sum / samples;
        float variance = (sumSquares / samples) - (mean * mean);
        float stdDev = (float) Math.sqrt(Math.max(0, variance));
        
        // Find most common prediction
        String mostCommon = "";
        int maxCount = 0;
        for (Map.Entry<String, Integer> entry : predictionCounts.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                mostCommon = entry.getKey();
            }
        }
        
        return new PredictionAnalysis(
            predictionCounts.size(), samples, min, max, mean, stdDev,
            mostCommon, maxCount
        );
    }
    
    private void evaluateModel(SimpleNetFloat model, ProductionSimulator sim) {
        // Test 1: Feature discrimination
        System.out.println("\n1. Feature Discrimination Tests:");
        
        // Test country discrimination
        System.out.println("\n  Country values:");
        for (String country : Arrays.asList("US", "UK", "CA", "BR", "IN", "OTHER")) {
            float avg = testFeatureValue(model, sim, "country", country, 100);
            System.out.printf("    %s: %.3f\n", country, avg);
        }
        
        // Test OS discrimination
        System.out.println("\n  OS values:");
        for (String os : Arrays.asList("ios", "android", "roku", "other")) {
            float avg = testFeatureValue(model, sim, "os", os, 100);
            System.out.printf("    %s: %.3f\n", os, avg);
        }
        
        // Test 2: High vs Low value apps
        System.out.println("\n2. App Value Discrimination:");
        float premiumAvg = 0, regularAvg = 0;
        
        // Test premium apps
        for (int i = 0; i < 100; i++) {
            Map<String, Object> request = sim.generateRequest();
            request.put("app_bundle", "com.premium.app" + (i % 10));
            premiumAvg += model.predictFloat(request);
        }
        premiumAvg /= 100;
        
        // Test regular apps
        for (int i = 0; i < 100; i++) {
            Map<String, Object> request = sim.generateRequest();
            request.put("app_bundle", "com.regular.app" + (i % 10));
            regularAvg += model.predictFloat(request);
        }
        regularAvg /= 100;
        
        System.out.printf("  Premium apps avg: %.3f\n", premiumAvg);
        System.out.printf("  Regular apps avg: %.3f\n", regularAvg);
        System.out.printf("  Difference: %.3f\n", premiumAvg - regularAvg);
        
        // Test 3: Overall prediction distribution
        System.out.println("\n3. Overall Prediction Distribution:");
        PredictionAnalysis analysis = analyzePredictions(model, sim, 10_000);
        analysis.printDetailed();
        
        // Assertions
        assertTrue(analysis.uniqueCount > 100, 
            "Model should produce diverse predictions, got " + analysis.uniqueCount);
        assertTrue(analysis.stdDev > 0.1f, 
            "Predictions should have meaningful variance, got " + analysis.stdDev);
        assertTrue(analysis.mostCommonPercent < 50, 
            "No single prediction should dominate, " + analysis.mostCommon + 
            " appears " + analysis.mostCommonPercent + "% of the time");
    }
    
    private float testFeatureValue(SimpleNetFloat model, ProductionSimulator sim, 
                                  String feature, Object value, int samples) {
        float sum = 0;
        for (int i = 0; i < samples; i++) {
            Map<String, Object> request = sim.generateRequest();
            request.put(feature, value);
            sum += model.predictFloat(request);
        }
        return sum / samples;
    }
    
    // Helper classes
    
    private static class ProductionSimulator {
        final Random rand = new Random(42);
        final String[] countries = {"US", "UK", "CA", "AU", "DE", "FR", "JP", "BR", "IN", "MX",
                                   "IT", "ES", "NL", "SE", "NO", "DK", "FI", "PL", "RU", "OTHER"};
        final String[] deviceTypes = {"phone", "tablet", "tv", "desktop", "other"};
        final String[] osTypes = {"ios", "android", "roku", "other"};
        
        Map<String, Object> generateRequest() {
            return generateRequest(true);
        }
        
        Map<String, Object> generateRequest(boolean allFeatures) {
            Map<String, Object> request = new HashMap<>();
            
            if (allFeatures) {
                // High cardinality features
                request.put("app_bundle", "com.app." + rand.nextInt(50_000));
                request.put("domain", "site" + rand.nextInt(10_000) + ".com");
                request.put("zone_id", rand.nextInt(4000));
                
                // Low cardinality features
                request.put("country", selectCountry());
                request.put("device_type", deviceTypes[rand.nextInt(deviceTypes.length)]);
                request.put("os", selectOS());
                
                // Continuous features
                request.put("bid_floor", 0.01f + rand.nextFloat() * 0.49f);
                request.put("hour_of_day", (float) rand.nextInt(24));
            } else {
                // Minimal feature set for testHighNegativeRatioTraining
                request.put("app_bundle", "com.app." + rand.nextInt(50_000));
                request.put("zone_id", rand.nextInt(4000));
                request.put("country", selectCountry());
                request.put("os", selectOS());
                request.put("bid_floor", 0.01f + rand.nextFloat() * 0.49f);
            }
            
            return request;
        }
        
        String selectCountry() {
            // 50% US, 20% tier-1, 30% rest
            float roll = rand.nextFloat();
            if (roll < 0.5f) return "US";
            if (roll < 0.7f) return countries[1 + rand.nextInt(9)]; // UK through MX
            return countries[10 + rand.nextInt(10)]; // Rest
        }
        
        String selectOS() {
            // 40% iOS, 50% Android, 10% other
            float roll = rand.nextFloat();
            if (roll < 0.4f) return "ios";
            if (roll < 0.9f) return "android";
            return rand.nextBoolean() ? "roku" : "other";
        }
        
        boolean shouldBid(Map<String, Object> request) {
            // Premium signals increase bid probability
            String country = (String) request.get("country");
            String os = (String) request.get("os");
            float bidFloor = (Float) request.get("bid_floor");
            
            float bidProb = 0.015f; // Base 1.5%
            
            if ("US".equals(country) || "UK".equals(country)) bidProb *= 1.5f;
            if ("ios".equals(os)) bidProb *= 1.3f;
            if (bidFloor < 0.1f) bidProb *= 1.2f;
            
            return rand.nextFloat() < bidProb;
        }
        
        float getBidValue(Map<String, Object> request) {
            String country = (String) request.get("country");
            String os = (String) request.get("os");
            float bidFloor = (Float) request.get("bid_floor");
            
            float base = 0.5f + rand.nextFloat() * 2.5f;
            
            // Adjust based on features
            if ("US".equals(country)) base *= 1.2f;
            else if ("UK".equals(country) || "CA".equals(country)) base *= 1.1f;
            else if ("BR".equals(country) || "IN".equals(country)) base *= 0.5f;
            
            if ("ios".equals(os)) base *= 1.15f;
            else if ("android".equals(os)) base *= 0.9f;
            
            return Math.max(base, bidFloor);
        }
    }
    
    private static class TrainingExample {
        final Map<String, Object> features;
        final float target;
        
        TrainingExample(Map<String, Object> features, float target) {
            this.features = features;
            this.target = target;
        }
    }
    
    private static class PredictionAnalysis {
        final int uniqueCount;
        final int totalSamples;
        final float min, max, mean, stdDev;
        final String mostCommon;
        final int mostCommonCount;
        final float mostCommonPercent;
        
        PredictionAnalysis(int uniqueCount, int totalSamples, float min, float max,
                          float mean, float stdDev, String mostCommon, int mostCommonCount) {
            this.uniqueCount = uniqueCount;
            this.totalSamples = totalSamples;
            this.min = min;
            this.max = max;
            this.mean = mean;
            this.stdDev = stdDev;
            this.mostCommon = mostCommon;
            this.mostCommonCount = mostCommonCount;
            this.mostCommonPercent = 100.0f * mostCommonCount / totalSamples;
        }
        
        boolean hasCollapsed() {
            return uniqueCount < 10 || 
                   stdDev < 0.01f || 
                   mostCommonPercent > 80;
        }
        
        String getCollapseReason() {
            if (uniqueCount < 10) return "Only " + uniqueCount + " unique predictions";
            if (stdDev < 0.01f) return "No variance (stdDev=" + stdDev + ")";
            if (mostCommonPercent > 80) return mostCommon + " appears in " + mostCommonPercent + "% of predictions";
            return "Unknown";
        }
        
        void print() {
            System.out.printf("  Predictions: %d unique, range [%.3f, %.3f], mean=%.3f, std=%.3f\n",
                uniqueCount, min, max, mean, stdDev);
            System.out.printf("  Most common: %s (%.1f%%)\n", mostCommon, mostCommonPercent);
        }
        
        void printDetailed() {
            System.out.printf("  Unique predictions: %d/%d\n", uniqueCount, totalSamples);
            System.out.printf("  Range: [%.3f, %.3f]\n", min, max);
            System.out.printf("  Mean: %.3f\n", mean);
            System.out.printf("  Std Dev: %.3f\n", stdDev);
            System.out.printf("  Most common: %s appears %d times (%.1f%%)\n", 
                mostCommon, mostCommonCount, mostCommonPercent);
        }
    }
}