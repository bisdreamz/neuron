package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Test different learning rates to prevent collapse when training
 * with auction penalties on every request.
 */
public class LearningRateCollapseTest {
    
    @Test
    public void testLearningRatesWithAuctionPenalties() {
        System.out.println("=== LEARNING RATES WITH AUCTION PENALTIES ===\n");
        System.out.println("Testing collapse with auction cost penalties...\n");
        
        // Test different learning rates
        float[] learningRates = {0.01f, 0.001f, 0.0001f, 0.00001f};
        
        for (float lr : learningRates) {
            testWithAuctionPenalty(lr);
        }
    }
    
    private void testWithAuctionPenalty(float lr) {
        System.out.printf("\n=== Testing LR = %f ===\n", lr);
        
        // Simplified features for faster testing
        Feature[] features = {
            Feature.embeddingLRU(1000, 32, "ZONEID"),
            Feature.embeddingLRU(1000, 32, "DOMAIN"),
            Feature.passthrough("BIDFLOOR")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(lr, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.hiddenDenseRelu(64))
            .withGlobalGradientClipping(1.0f)
            .output(Layers.outputHuberRegression(1, optimizer, 3.0f));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Define premium segments (5%)
        Set<String> premiumSegments = new HashSet<>();
        for (int i = 0; i < 50; i++) {
            premiumSegments.add(i + "_" + i);
        }
        
        // Track predictions over time
        List<Float> premiumPredHistory = new ArrayList<>();
        List<Float> regularPredHistory = new ArrayList<>();
        boolean collapsed = false;
        
        // Train for 20k steps (faster test)
        for (int step = 0; step < 20_000; step++) {
            int zoneId = rand.nextInt(1000);
            int domainId = rand.nextInt(1000);
            String segment = zoneId + "_" + domainId;
            
            Map<String, Object> input = Map.of(
                "ZONEID", zoneId,
                "DOMAIN", domainId,
                "BIDFLOOR", 1.0f
            );
            
            // Step 1: Auction penalty (every request)
            float penalty = -0.01f - rand.nextFloat() * 0.04f; // -$0.01 to -$0.05
            model.train(input, penalty);
            
            // Step 2: Bid result
            boolean isPremium = premiumSegments.contains(segment);
            float bidValue;
            
            if (isPremium && rand.nextFloat() < 0.8f) {
                bidValue = 2.0f + rand.nextFloat(); // $2-3 for premium
            } else if (!isPremium && rand.nextFloat() < 0.05f) {
                bidValue = 0.2f + rand.nextFloat() * 0.3f; // $0.2-0.5 for regular
            } else {
                bidValue = 0.0f; // No bid
            }
            
            model.train(input, bidValue);
            
            // Monitor every 2000 steps
            if (step > 0 && step % 2000 == 0) {
                // Test premium segment
                float premiumPred = model.predictFloat(Map.of(
                    "ZONEID", 10, "DOMAIN", 10, "BIDFLOOR", 1.0f));
                
                // Test regular segment
                float regularPred = model.predictFloat(Map.of(
                    "ZONEID", 500, "DOMAIN", 500, "BIDFLOOR", 1.0f));
                
                premiumPredHistory.add(premiumPred);
                regularPredHistory.add(regularPred);
                
                System.out.printf("Step %5d: Premium=$%.3f, Regular=$%.3f, Diff=$%.3f\n",
                    step, premiumPred, regularPred, premiumPred - regularPred);
                
                // Check for collapse
                if (Math.abs(premiumPred - regularPred) < 0.01f) {
                    System.out.println("⚠️ COLLAPSED - Predictions converged!");
                    collapsed = true;
                    break;
                }
            }
        }
        
        // Analyze stability
        if (premiumPredHistory.size() >= 2) {
            float finalPremium = premiumPredHistory.get(premiumPredHistory.size() - 1);
            float finalRegular = regularPredHistory.get(regularPredHistory.size() - 1);
            float finalDiff = Math.abs(finalPremium - finalRegular);
            
            boolean stable = !collapsed && finalDiff > 0.5f && finalPremium > 0.5f;
            
            System.out.printf("\nFinal state: Premium=$%.3f, Regular=$%.3f, Diff=$%.3f\n",
                finalPremium, finalRegular, finalDiff);
            System.out.println(stable ? "✓ STABLE - Model differentiates segments" :
                                      "❌ UNSTABLE - Model cannot differentiate");
        }
    }
    
    @Test
    public void testSimpleCaseWithoutPenalty() {
        System.out.println("\n=== SIMPLE CASE WITHOUT PENALTY ===\n");
        System.out.println("Testing if model works without auction penalties...\n");
        
        testSimpleCase(0.001f, false);
        testSimpleCase(0.001f, true);
    }
    
    private void testSimpleCase(float lr, boolean withPenalty) {
        System.out.printf("\nTesting LR=%f %s penalties\n", lr, withPenalty ? "WITH" : "WITHOUT");
        
        Feature[] features = {
            Feature.embeddingLRU(100, 16, "zone"),
            Feature.passthrough("value")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(lr, 0.0f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(64))
            .withGlobalGradientClipping(1.0f)
            .output(Layers.outputHuberRegression(1, optimizer, 3.0f));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Define zone CPMs
        Map<Integer, Float> zoneCPMs = new HashMap<>();
        for (int i = 0; i < 10; i++) {
            zoneCPMs.put(i, i < 5 ? 2.0f : 0.5f); // First 5 zones are premium
        }
        
        // Train
        for (int step = 0; step < 10_000; step++) {
            int zone = rand.nextInt(10);
            Map<String, Object> input = Map.of("zone", zone, "value", 1.0f);
            
            if (withPenalty) {
                // First train with penalty
                model.train(input, -0.03f);
            }
            
            // Then train with actual value
            float target = zoneCPMs.get(zone);
            model.train(input, target);
        }
        
        // Test
        float premiumPred = model.predictFloat(Map.of("zone", 0, "value", 1.0f));
        float regularPred = model.predictFloat(Map.of("zone", 7, "value", 1.0f));
        
        System.out.printf("Premium zone: $%.3f (expected $2.00)\n", premiumPred);
        System.out.printf("Regular zone: $%.3f (expected $0.50)\n", regularPred);
        System.out.printf("Difference: $%.3f\n", premiumPred - regularPred);
        
        boolean success = Math.abs(premiumPred - regularPred) > 1.0f;
        System.out.println(success ? "✓ SUCCESS" : "❌ FAILED");
    }
    
    private void testLearningRate(String name, float lr) {
        System.out.println("--- " + name + " (LR=" + lr + ") ---");
        
        // Simplest possible network - embedding directly to output
        Feature[] features = {Feature.embeddingLRU(100, 8, "item")};
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(new SgdOptimizer(lr))
            .layer(Layers.inputMixed(features))
            .output(Layers.outputLinearRegression(1));  // No hidden layers
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        int trainedSamples = 0;
        float totalLoss = 0;
        
        // Train for limited steps and monitor loss
        for (int i = 0; i < 500; i++) {
            boolean isGood = rand.nextBoolean();
            Map<String, Object> input = Map.of("item", isGood ? "good_" + rand.nextInt(10) : "bad_" + rand.nextInt(10));
            float target = isGood ? 1.0f : 0.0f;
            
            // Get prediction before training to calculate loss
            float pred = model.predictFloat(input);
            float loss = (pred - target) * (pred - target);
            totalLoss += loss;
            
            model.train(input, target);
            trainedSamples++;
            
            // Check for NaN or infinite values
            float newPred = model.predictFloat(input);
            if (Float.isNaN(newPred) || Float.isInfinite(newPred)) {
                System.out.printf("ERROR: NaN/Infinite prediction at step %d\\n", i);
                break;
            }
        }
        
        float avgLoss = totalLoss / trainedSamples;
        System.out.printf("Average loss: %.6f\\n", avgLoss);
        
        // Test for collapse
        Set<String> uniquePreds = new HashSet<>();
        float goodSum = 0, badSum = 0;
        
        for (int i = 0; i < 20; i++) {
            Map<String, Object> input = Map.of("item", i < 10 ? "good_0" : "bad_0");
            float pred = model.predictFloat(input);
            uniquePreds.add(String.format("%.4f", pred));
            
            if (i < 10) goodSum += pred;
            else badSum += pred;
        }
        
        float goodAvg = goodSum / 10;
        float badAvg = badSum / 10;
        boolean collapsed = uniquePreds.size() < 3;
        
        System.out.printf("Unique predictions: %d\\n", uniquePreds.size());
        System.out.printf("Good avg: %.4f, Bad avg: %.4f\\n", goodAvg, badAvg);
        System.out.printf("Discrimination: %.4f\\n", Math.abs(goodAvg - badAvg));
        System.out.printf("Result: %s\\n\\n", collapsed ? "⚠️ COLLAPSED" : "✓ LEARNING");
    }
    
    private void testTargetScale(String name, float minTarget, float maxTarget) {
        System.out.println("--- " + name + " (targets " + minTarget + " to " + maxTarget + ") ---");
        
        Feature[] features = {Feature.embeddingLRU(50, 4, "item")};
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(new SgdOptimizer(0.001f))  // Use low LR
            .layer(Layers.inputMixed(features))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        float totalLoss = 0;
        int trainedSamples = 0;
        
        for (int i = 0; i < 500; i++) {
            boolean isGood = rand.nextBoolean();
            Map<String, Object> input = Map.of("item", isGood ? "good_" + rand.nextInt(10) : "bad_" + rand.nextInt(10));
            float target = isGood ? 
                (minTarget + rand.nextFloat() * (maxTarget - minTarget)) : 
                minTarget;
                
            float pred = model.predictFloat(input);
            float loss = (pred - target) * (pred - target);
            totalLoss += loss;
            
            model.train(input, target);
            trainedSamples++;
        }
        
        float avgLoss = totalLoss / trainedSamples;
        System.out.printf("Average loss: %.6f\\n", avgLoss);
        
        // Test for collapse
        Set<String> uniquePreds = new HashSet<>();
        for (int i = 0; i < 20; i++) {
            Map<String, Object> input = Map.of("item", i < 10 ? "good_0" : "bad_0");
            float pred = model.predictFloat(input);
            uniquePreds.add(String.format("%.4f", pred));
        }
        
        boolean collapsed = uniquePreds.size() < 3;
        System.out.printf("Unique predictions: %d\\n", uniquePreds.size());
        System.out.printf("Result: %s\\n\\n", collapsed ? "⚠️ COLLAPSED" : "✓ LEARNING");
    }
}