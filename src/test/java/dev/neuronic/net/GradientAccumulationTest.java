package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Test to diagnose gradient accumulation and learning issues with MixedFeatureInputLayer.
 * This test verifies:
 * 1. One-hot only features cannot learn (no parameters)
 * 2. Embeddings need optimizer.forEmbeddings() for proper learning
 * 3. Passthrough features can learn properly
 */
public class GradientAccumulationTest {
    
    @Test
    public void testOneHotOnlyCannotLearn() {
        System.out.println("=== TEST 1: One-Hot Only (No Learnable Parameters) ===");
        System.out.println("This mimics CorrectProductionScenarioTest configuration\n");
        
        // Just like CorrectProductionScenarioTest - all one-hot features
        Feature[] features = {
            Feature.oneHot(4, "OS"),
            Feature.oneHot(100, "ZONEID"),
            Feature.oneHot(100, "DOMAIN"),
            Feature.oneHot(100, "PUB")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Train with clear pattern
        System.out.println("Training 1000 samples with pattern: zone < 50 → 0.0, zone >= 50 → 1.0");
        for (int i = 0; i < 1000; i++) {
            int zone = rand.nextInt(100);
            float target = zone < 50 ? 0.0f : 1.0f;
            
            Map<String, Object> input = Map.of(
                "OS", rand.nextInt(4),
                "ZONEID", zone,
                "DOMAIN", rand.nextInt(100),
                "PUB", rand.nextInt(100)
            );
            
            model.train(input, target);
        }
        
        // Test predictions
        System.out.println("\nTesting predictions:");
        Set<String> uniquePreds = new HashSet<>();
        float lowSum = 0, highSum = 0;
        
        for (int zone = 0; zone < 100; zone += 10) {
            float pred = model.predictFloat(Map.of(
                "OS", 0, "ZONEID", zone, "DOMAIN", 0, "PUB", 0
            ));
            uniquePreds.add(String.format("%.4f", pred));
            
            if (zone < 50) lowSum += pred;
            else highSum += pred;
            
            if (zone % 20 == 0) {
                System.out.printf("  Zone %d → %.4f (expected %.1f)\n", 
                    zone, pred, zone < 50 ? 0.0f : 1.0f);
            }
        }
        
        float lowAvg = lowSum / 5;
        float highAvg = highSum / 5;
        
        System.out.printf("\nLow zones (0-40) avg: %.4f\n", lowAvg);
        System.out.printf("High zones (50-90) avg: %.4f\n", highAvg);
        System.out.printf("Difference: %.4f\n", highAvg - lowAvg);
        System.out.printf("Unique predictions: %d\n", uniquePreds.size());
        
        boolean canLearn = uniquePreds.size() > 5 && Math.abs(highAvg - lowAvg) > 0.1f;
        System.out.println("\nResult: " + (canLearn ? "✓ CAN LEARN" : "❌ CANNOT LEARN"));
        System.out.println("EXPECTED: Cannot learn - one-hot features have no learnable parameters!\n");
    }
    
    @Test
    public void testPassthroughCanLearn() {
        System.out.println("=== TEST 2: Passthrough Features (Can Learn) ===\n");
        
        // Mix of one-hot and passthrough features
        Feature[] features = {
            Feature.oneHot(4, "OS"),
            Feature.passthrough("ZONE_SCORE"),  // This can learn!
            Feature.passthrough("DOMAIN_SCORE"), // This can learn!
            Feature.passthrough("PUB_SCORE")     // This can learn!
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Train with clear pattern based on zone_score
        System.out.println("Training 1000 samples with pattern: zone_score < 0.5 → 0.0, >= 0.5 → 1.0");
        for (int i = 0; i < 1000; i++) {
            float zoneScore = rand.nextFloat();
            float target = zoneScore < 0.5f ? 0.0f : 1.0f;
            
            Map<String, Object> input = Map.of(
                "OS", rand.nextInt(4),
                "ZONE_SCORE", zoneScore,
                "DOMAIN_SCORE", rand.nextFloat(),
                "PUB_SCORE", rand.nextFloat()
            );
            
            model.train(input, target);
        }
        
        // Test predictions
        System.out.println("\nTesting predictions:");
        Set<String> uniquePreds = new HashSet<>();
        
        for (float zoneScore = 0.0f; zoneScore <= 1.0f; zoneScore += 0.1f) {
            float pred = model.predictFloat(Map.of(
                "OS", 0,
                "ZONE_SCORE", zoneScore,
                "DOMAIN_SCORE", 0.5f,
                "PUB_SCORE", 0.5f
            ));
            uniquePreds.add(String.format("%.4f", pred));
            
            System.out.printf("  Zone score %.1f → %.4f (expected %.1f)\n", 
                zoneScore, pred, zoneScore < 0.5f ? 0.0f : 1.0f);
        }
        
        System.out.printf("\nUnique predictions: %d\n", uniquePreds.size());
        
        boolean canLearn = uniquePreds.size() > 5;
        System.out.println("Result: " + (canLearn ? "✓ CAN LEARN" : "❌ CANNOT LEARN"));
        System.out.println("EXPECTED: Can learn - passthrough features flow through network!\n");
    }
    
    @Test
    public void testEmbeddingsNeedSpecialOptimizer() {
        System.out.println("=== TEST 3: Embeddings (Need optimizer.forEmbeddings()) ===\n");
        
        // Using embeddings instead of one-hot
        Feature[] features = {
            Feature.embedding(100, 16, "ZONEID"),   // 100 zones, 16-dim embeddings
            Feature.embedding(100, 16, "DOMAIN"),   // 100 domains, 16-dim embeddings
            Feature.oneHot(4, "OS"),
            Feature.passthrough("BIDFLOOR")
        };
        
        // Low learning rate to simulate the issue
        AdamWOptimizer optimizer = new AdamWOptimizer(0.0001f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(64))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Train with zone-based pattern
        System.out.println("Training with low LR (0.0001) - embeddings need 5x higher!");
        for (int i = 0; i < 2000; i++) {
            int zone = rand.nextInt(100);
            float target = zone < 50 ? 0.1f : 0.9f;
            
            Map<String, Object> input = Map.of(
                "ZONEID", zone,
                "DOMAIN", rand.nextInt(100),
                "OS", rand.nextInt(4),
                "BIDFLOOR", rand.nextFloat()
            );
            
            model.train(input, target);
            
            if (i % 500 == 499) {
                // Test current performance
                float lowPred = model.predictFloat(Map.of(
                    "ZONEID", 10, "DOMAIN", 10, "OS", 0, "BIDFLOOR", 0.5f
                ));
                float highPred = model.predictFloat(Map.of(
                    "ZONEID", 80, "DOMAIN", 80, "OS", 0, "BIDFLOOR", 0.5f
                ));
                System.out.printf("  Step %d: Low zone → %.4f, High zone → %.4f, Diff → %.4f\n",
                    i + 1, lowPred, highPred, highPred - lowPred);
            }
        }
        
        // Final test
        System.out.println("\nFinal predictions:");
        Set<String> uniquePreds = new HashSet<>();
        float lowSum = 0, highSum = 0;
        
        for (int zone = 0; zone < 100; zone += 10) {
            float pred = model.predictFloat(Map.of(
                "ZONEID", zone, "DOMAIN", 50, "OS", 0, "BIDFLOOR", 0.5f
            ));
            uniquePreds.add(String.format("%.4f", pred));
            
            if (zone < 50) lowSum += pred;
            else highSum += pred;
        }
        
        float lowAvg = lowSum / 5;
        float highAvg = highSum / 5;
        
        System.out.printf("\nLow zones avg: %.4f\n", lowAvg);
        System.out.printf("High zones avg: %.4f\n", highAvg);
        System.out.printf("Difference: %.4f\n", highAvg - lowAvg);
        System.out.printf("Unique predictions: %d\n", uniquePreds.size());
        
        boolean learned = Math.abs(highAvg - lowAvg) > 0.2f;
        System.out.println("\nResult: " + (learned ? "✓ LEARNED WELL" : "⚠️ WEAK LEARNING"));
        System.out.println("EXPECTED: Weak learning - embeddings need optimizer.forEmbeddings()!");
        System.out.println("Embeddings should get 5x LR (0.0005) but are using base LR (0.0001)\n");
    }
    
    @Test
    public void diagnoseCorrectProductionScenario() {
        System.out.println("=== DIAGNOSIS: Why CorrectProductionScenarioTest Fails ===\n");
        
        System.out.println("CorrectProductionScenarioTest configuration:");
        System.out.println("- Feature.oneHot(50, \"OS\")");
        System.out.println("- Feature.oneHot(1000, \"ZONEID\")");
        System.out.println("- Feature.oneHot(1000, \"DOMAIN\")");
        System.out.println("- Feature.oneHot(1000, \"PUB\")");
        System.out.println("- Feature.autoScale(0f, 20f, \"BIDFLOOR\")\n");
        
        System.out.println("PROBLEM: One-hot encoding has NO learnable parameters!");
        System.out.println("- One-hot just sets 1.0 at the category index");
        System.out.println("- No weights to update during backpropagation");
        System.out.println("- Only the hidden layers can learn, but they see sparse inputs\n");
        
        System.out.println("SOLUTION: Use embeddings for high-cardinality features:");
        System.out.println("- Feature.embedding(1000, 32, \"ZONEID\")  // 32-dim dense vectors");
        System.out.println("- Feature.embedding(1000, 32, \"DOMAIN\")");
        System.out.println("- Feature.embedding(1000, 32, \"PUB\")");
        System.out.println("- Feature.oneHot(50, \"OS\")  // OK for low cardinality");
        System.out.println("- Feature.autoScale(0f, 20f, \"BIDFLOOR\")\n");
        
        System.out.println("With embeddings:");
        System.out.println("- Each category gets a learnable dense vector");
        System.out.println("- Gradients update these vectors during training");
        System.out.println("- Model can learn relationships between categories");
        System.out.println("- BUT: MixedFeatureInputLayer needs to use optimizer.forEmbeddings()!");
    }
}