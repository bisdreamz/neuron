package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamOptimizer;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.Optimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * CORRECT production scenario matching what you actually do:
 * 
 * 1. EVERY REQUEST gets penalty training (auction cost)
 * 2. ~50% of requests ALSO get bid training (some win, some get 0)
 * 3. Premium segments (5%) get MORE traffic AND higher bid rates
 * 4. Overall: 100,000 penalty trainings, ~50,000 bid trainings
 */
public class SerialCorrectProductionScenarioTest {
    
    @Test
    public void testCorrectProductionScenario() {
        System.out.println("=== CORRECT PRODUCTION SCENARIO ===\n");
        System.out.println("What actually happens:");
        System.out.println("1. EVERY request → penalty training");
        System.out.println("2. ~50% of requests → ALSO bid training");
        System.out.println("3. Premium segments get MORE traffic");
        System.out.println("4. Overall ratio: 2:1 penalties:bids\n");
        
        // Test with different penalty approaches
        testWithPenalty("Negative penalty (-$0.0003)", -0.0003f);
        testWithPenalty("Tiny positive ($0.0001)", 0.0001f);
        testWithPenalty("Very tiny positive ($0.00001)", 0.00001f);
    }
    
    private void testWithPenalty(String name, float penaltyValue) {
        System.out.printf("\n=== %s ===\n", name);

        Feature[] features = {
            Feature.oneHot(2, "OS"), // ios, android
            Feature.embeddingLRU(10000, 32, "ZONEID"),
            Feature.hashedEmbedding(5000, 16, "DOMAIN"),
            Feature.embeddingLRU(2000, 16, "PUB"),
            Feature.autoScale(0f, 20f, "BIDFLOOR")
        };

        //Optimizer optimizer = new AdamWOptimizer(0.002f, 0.00001f);
        Optimizer optimizer = new SgdOptimizer(0.001f);

        NeuralNet net = NeuralNet.newBuilder()
                .setDefaultOptimizer(optimizer)
                .layer(Layers.inputMixed(features))
                .layer(Layers.hiddenDenseLeakyRelu(512))
                .layer(Layers.hiddenDenseLeakyRelu(256))
                .layer(Layers.hiddenDenseLeakyRelu(128))
                .layer(Layers.hiddenDenseLeakyRelu(64))
                .output(Layers.outputLinearRegression(1));

        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);

        Random rand = new Random(42);

        // Define premium feature sets
        Set<String> premiumZones = new HashSet<>();
        for (int i = 0; i < 500; i++) premiumZones.add("zone_" + i);

        Set<String> premiumPubs = new HashSet<>();
        for (int i = 0; i < 100; i++) premiumPubs.add("pub_" + i);

        Set<String> premiumDomains = new HashSet<>();
        for (int i = 0; i < 250; i++) premiumDomains.add("domain_" + i + ".com");

        int totalPenaltyTrains = 0;
        int totalBidTrains = 0;

        int steps = 20000;
        System.out.println("Simulating " + steps + " requests with realistic, interleaved data...");

        for (int step = 0; step < steps; step++) {
            String os = rand.nextBoolean() ? "ios" : "android";
            String zoneId = "zone_" + rand.nextInt(10000);
            String pubId = "pub_" + rand.nextInt(2000);
            String domain = "domain_" + rand.nextInt(5000) + ".com";
            float bidfloor = 0.1f + rand.nextFloat() * 4.0f;

            Map<String, Object> input = Map.of(
                "OS", os,
                "ZONEID", zoneId,
                "DOMAIN", domain,
                "PUB", pubId,
                "BIDFLOOR", bidfloor
            );

            // Always train with penalty
            model.train(input, penaltyValue);
            totalPenaltyTrains++;

            // Determine bid value based on feature combinations
            int score = 0;
            if (premiumZones.contains(zoneId)) score++;
            if (premiumPubs.contains(pubId)) score++;
            if (premiumDomains.contains(domain)) score++;
            if (os.equals("ios")) score++;

            float bidRate = 0.05f + (score * 0.20f); // Higher score = higher bid rate
            float bidValue = 0.0f;

            if (rand.nextFloat() < bidRate) {
                bidValue = 0.5f + (score * 0.75f) + (rand.nextFloat() * score);
                model.train(input, bidValue);
                totalBidTrains++;
            }
        
            if (step > 0 && step % 2000 == 0) {
                System.out.printf("Step %d: %d penalties, %d bids (ratio %.1f:1)\n",
                    step, totalPenaltyTrains, totalBidTrains,
                    totalBidTrains > 0 ? (float)totalPenaltyTrains / totalBidTrains : 0);
                
                // Test predictions on high-value and low-value segments
                Map<String, Object> premiumInput = Map.of("OS", "ios", "ZONEID", "zone_1", "DOMAIN", "domain_1.com", "PUB", "pub_1", "BIDFLOOR", 3.0f);
                Map<String, Object> regularInput = Map.of("OS", "android", "ZONEID", "zone_9999", "DOMAIN", "domain_4999.com", "PUB", "pub_1999", "BIDFLOOR", 0.2f);
                float premiumPred = model.predictFloat(premiumInput);
                float regularPred = model.predictFloat(regularInput);
                
                System.out.printf("  High-Value Prediction: $%.3f, Low-Value Prediction: $%.3f\n", premiumPred, regularPred);
            }
        }
        
        System.out.printf("\nFinal: %d penalty trains, %d bid trains (ratio %.1f:1)\n",
            totalPenaltyTrains, totalBidTrains,
            (float)totalPenaltyTrains / totalBidTrains);
        
        System.out.println("\nTesting prediction accuracy on a spectrum of feature combinations:");
        evaluatePredictions(model, premiumZones, premiumPubs, premiumDomains);
    }
    
    private void evaluatePredictions(SimpleNetFloat model, Set<String> premiumZones, Set<String> premiumPubs, Set<String> premiumDomains) {
        // Test a variety of combinations
        Map<String, Map<String, Object>> testCases = new LinkedHashMap<>();
        testCases.put("Fully Premium", Map.of("OS", "ios", "ZONEID", "zone_1", "DOMAIN", "domain_1.com", "PUB", "pub_1", "BIDFLOOR", 3.0f));
        testCases.put("Mostly Premium", Map.of("OS", "ios", "ZONEID", "zone_2", "DOMAIN", "domain_2.com", "PUB", "pub_1500", "BIDFLOOR", 2.5f));
        testCases.put("Mixed (Premium Zone)", Map.of("OS", "android", "ZONEID", "zone_3", "DOMAIN", "domain_4000.com", "PUB", "pub_1600", "BIDFLOOR", 1.0f));
        testCases.put("Mixed (Premium OS)", Map.of("OS", "ios", "ZONEID", "zone_8000", "DOMAIN", "domain_4001.com", "PUB", "pub_1700", "BIDFLOOR", 1.2f));
        testCases.put("Mostly Regular", Map.of("OS", "android", "ZONEID", "zone_9000", "DOMAIN", "domain_4500.com", "PUB", "pub_50", "BIDFLOOR", 0.5f));
        testCases.put("Fully Regular", Map.of("OS", "android", "ZONEID", "zone_9999", "DOMAIN", "domain_4999.com", "PUB", "pub_1999", "BIDFLOOR", 0.2f));

        List<Float> predictions = new ArrayList<>();
        Set<String> uniquePredictions = new HashSet<>();
        
        System.out.println("--- Predictions ---");
        for (Map.Entry<String, Map<String, Object>> entry : testCases.entrySet()) {
            float pred = model.predictFloat(entry.getValue());
            predictions.add(pred);
            uniquePredictions.add(String.format("%.4f", pred));
            System.out.printf("%-20s: $%.4f\n", entry.getKey(), pred);
        }

        // Success criteria:
        // 1. Predictions should be ordered from high to low, matching the segment value.
        // 2. There should be significant differentiation between the top and bottom segments.
        // 3. The model should not collapse to a single prediction.
        
        boolean orderedCorrectly = true;
        for (int i = 0; i < predictions.size() - 1; i++) {
            if (predictions.get(i) < predictions.get(i + 1)) {
                orderedCorrectly = false;
                break;
            }
        }

        float differentiation = predictions.getFirst() - predictions.getLast();
        boolean hasDifferentiation = differentiation > 1.0f;
        boolean hasDiversity = uniquePredictions.size() > 3;

        System.out.println("\n--- Evaluation ---");
        System.out.printf("Predictions are correctly ordered: %s\n", orderedCorrectly);
        System.out.printf("Differentiation (Top - Bottom): $%.4f\n", differentiation);
        System.out.printf("Prediction Diversity: %d unique values\n", uniquePredictions.size());

        boolean success = orderedCorrectly && hasDifferentiation && hasDiversity;
        System.out.println(success ? "\n[SUCCESS] - Model has learned complex feature interactions!" : "\n[FAILURE] - Model has not learned the pattern correctly.");
    }
    
    private record PredictionResult(float premiumPred, float regularPred, int step) {}
}