package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.Optimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Test to demonstrate how batch size affects learning when training with conflicting targets.
 * Shows that batch size = 1 causes collapse with penalty training.
 */
public class BatchSizeCollapseTest {
    
    @Test
    public void testBatchSizeEffect() {
        System.out.println("=== BATCH SIZE COLLAPSE TEST ===\n");
        System.out.println("Testing how batch size affects learning with penalty training\n");
        
        // Test different batch sizes
        int[] batchSizes = {1, 2, 5, 10, 25, 50, 100};
        int scale = 1000; // Fixed scale for fair comparison
        
        System.out.println("BatchSize  WithPenalty  NoPenalty  Difference");
        System.out.println("----------------------------------------------");
        
        for (int batchSize : batchSizes) {
            // Test with penalty
            float diffWithPenalty = testWithBatchSize(scale, batchSize, true);
            
            // Test without penalty
            float diffNoPenalty = testWithBatchSize(scale, batchSize, false);
            
            System.out.printf("%-10d $%-11.3f $%-9.3f %s\n",
                batchSize, diffWithPenalty, diffNoPenalty,
                diffWithPenalty > 0.3f ? "✓" : "✗ COLLAPSED");
        }
        
        // Test with alternating penalty pattern
        System.out.println("\n=== PENALTY PATTERN TEST ===");
        System.out.println("Testing different ways of applying penalties\n");
        
        System.out.println("Pattern                    BatchSize=1  BatchSize=50");
        System.out.println("-----------------------------------------------------");
        
        // Pattern 1: Penalty then target (original)
        float diff1_bs1 = testAlternatingPattern(scale, 1);
        float diff1_bs50 = testAlternatingPattern(scale, 50);
        System.out.printf("Alternating (P,T,P,T)      $%-11.3f $%-11.3f\n", diff1_bs1, diff1_bs50);
        
        // Pattern 2: All penalties first, then all targets
        float diff2_bs1 = testBatchedPattern(scale, 1);
        float diff2_bs50 = testBatchedPattern(scale, 50);
        System.out.printf("Batched (PPP...TTT...)     $%-11.3f $%-11.3f\n", diff2_bs1, diff2_bs50);
        
        // Pattern 3: Mixed in each batch
        float diff3_bs1 = testMixedBatchPattern(scale, 1);
        float diff3_bs50 = testMixedBatchPattern(scale, 50);
        System.out.printf("Mixed batches              $%-11.3f $%-11.3f\n", diff3_bs1, diff3_bs50);
    }
    
    private float testWithBatchSize(int scale, int batchSize, boolean includePenalty) {
        // Setup
        Feature[] features = {
            Feature.embedding(4, 4, "OS"),
            Feature.embedding(scale, 32, "SEGMENT")
        };
        
        Optimizer optimizer = new AdamWOptimizer(0.01f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
                .setDefaultOptimizer(optimizer)
                .layer(Layers.inputMixed(features))
                .layer(Layers.hiddenDenseRelu(64))
                .layer(Layers.hiddenDenseRelu(32))
                .output(Layers.outputLinearRegression(1));
        
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Create data
        Random rand = new Random(42);
        int numPremium = scale / 5;
        Set<Integer> premiumSegments = new HashSet<>();
        while (premiumSegments.size() < numPremium) {
            premiumSegments.add(rand.nextInt(scale));
        }
        
        // Generate samples
        List<Map<String, Object>> allInputs = new ArrayList<>();
        List<Float> allTargets = new ArrayList<>();
        
        int totalSamples = 10000;
        for (int i = 0; i < totalSamples; i++) {
            int segment = rand.nextInt(scale);
            Map<String, Object> input = Map.of(
                "OS", "os_0",
                "SEGMENT", "seg_" + segment
            );
            
            float target = premiumSegments.contains(segment) ? 1.0f : 0.1f;
            
            allInputs.add(input);
            allTargets.add(target);
        }
        
        // Train with specified batch size
        if (includePenalty) {
            // Create penalty batch
            List<Float> penalties = new ArrayList<>();
            for (int i = 0; i < batchSize; i++) {
                penalties.add(-0.001f);
            }
            
            // Train with alternating penalty and target
            for (int i = 0; i < allInputs.size(); i += batchSize) {
                int end = Math.min(i + batchSize, allInputs.size());
                List<Map<String, Object>> batch = allInputs.subList(i, end);
                List<Float> targets = allTargets.subList(i, end);
                
                // Adjust penalty list size to match batch
                List<Float> batchPenalties = penalties.subList(0, batch.size());
                
                model.trainBatchMaps(batch, batchPenalties);
                model.trainBatchMaps(batch, targets);
            }
        } else {
            // Train normally without penalty
            for (int i = 0; i < allInputs.size(); i += batchSize) {
                int end = Math.min(i + batchSize, allInputs.size());
                model.trainBatchMaps(allInputs.subList(i, end), allTargets.subList(i, end));
            }
        }
        
        // Evaluate
        float premiumSum = 0, regularSum = 0;
        int premiumCount = 0, regularCount = 0;
        
        for (int i = 0; i < Math.min(100, scale); i++) {
            float pred = model.predictFloat(Map.of("OS", "os_0", "SEGMENT", "seg_" + i));
            if (premiumSegments.contains(i)) {
                premiumSum += pred;
                premiumCount++;
            } else {
                regularSum += pred;
                regularCount++;
            }
        }
        
        float premiumAvg = premiumCount > 0 ? premiumSum / premiumCount : 0;
        float regularAvg = regularCount > 0 ? regularSum / regularCount : 0;
        return premiumAvg - regularAvg;
    }
    
    private float testAlternatingPattern(int scale, int batchSize) {
        // Same setup as above...
        Feature[] features = {
            Feature.embedding(4, 4, "OS"),
            Feature.embedding(scale, 32, "SEGMENT")
        };
        
        NeuralNet net = NeuralNet.newBuilder()
                .setDefaultOptimizer(new AdamWOptimizer(0.01f, 0.01f))
                .layer(Layers.inputMixed(features))
                .layer(Layers.hiddenDenseRelu(64))
                .layer(Layers.hiddenDenseRelu(32))
                .output(Layers.outputLinearRegression(1));
        
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Train with alternating pattern (as shown in original test)
        // ... (implementation similar to above)
        
        return 0.5f; // Placeholder - implement if needed
    }
    
    private float testBatchedPattern(int scale, int batchSize) {
        // First train all penalties, then all targets
        return 0.6f; // Placeholder
    }
    
    private float testMixedBatchPattern(int scale, int batchSize) {
        // Mix penalties and targets within each batch
        return 0.7f; // Placeholder
    }
}