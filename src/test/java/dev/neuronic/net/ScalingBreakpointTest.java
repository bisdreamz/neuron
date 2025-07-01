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
 * Automated test to find where neural network learning breaks down as feature space scales up.
 * Tests multiple scales and learning rates to identify the breaking point.
 */
public class ScalingBreakpointTest {
    
    // Test configuration
    private static final int MIN_UNIQUE_VALUES = 10;      // Start small like MinimalDebugTest
    private static final int MAX_UNIQUE_VALUES = 20000;  // End at production scale
    private static final int NUM_INTERVALS = 10;         // Number of scale points to test
    private static final float[] LEARNING_RATES = {0.005f, 0.001f, 0.0005f, 0.0001f};
    private static final int TRAINING_STEPS = 100_000;      // Fixed steps for fair comparison
    private static final float SUCCESS_THRESHOLD = 0.7f; // Minimum differentiation to count as success
    private static final int BATCH_SIZE = 1;
    
    @Test
    public void findScalingBreakpoint() {
        System.out.println("=== SCALING BREAKPOINT ANALYSIS ===\n");
        System.out.println("Testing neural network learning across different feature space scales");
        System.out.println("Looking for where learning breaks down...\n");
        
        // Generate logarithmically spaced intervals
        List<Integer> scalePoints = generateLogScalePoints(MIN_UNIQUE_VALUES, MAX_UNIQUE_VALUES, NUM_INTERVALS);
        
        // Results tracking
        Map<String, List<TestResult>> results = new HashMap<>();
        for (float lr : LEARNING_RATES) {
            results.put(String.format("LR=%.4f", lr), new ArrayList<>());
        }
        
        // Test each scale point
        for (int scalePoint : scalePoints) {
            System.out.printf("\n=== TESTING SCALE: %d unique values per feature ===\n", scalePoint);
            
            // Calculate reasonable embedding dimensions based on scale
            int embeddingDim = calculateEmbeddingDim(scalePoint);
            int hiddenSize = calculateHiddenSize(scalePoint);
            
            System.out.printf("Config: %d unique values, %d-dim embeddings, %d hidden units\n", 
                scalePoint, embeddingDim, hiddenSize);
            
            // Test each learning rate at this scale
            for (float lr : LEARNING_RATES) {
                TestResult result = testAtScale(scalePoint, embeddingDim, hiddenSize, lr);
                results.get(String.format("LR=%.4f", lr)).add(result);
                
                System.out.printf("  LR=%.4f: Premium=$%.3f, Regular=$%.3f, Diff=$%.3f, Success=%s\n",
                    lr, result.premiumAvg, result.regularAvg, result.differentiation, 
                    result.success ? "YES" : "NO");
            }
        }
        
        // Print summary
        printSummary(scalePoints, results);
    }
    
    private List<Integer> generateLogScalePoints(int min, int max, int numPoints) {
        List<Integer> points = new ArrayList<>();
        
        // Use logarithmic scaling to cover the range better
        double logMin = Math.log10(min);
        double logMax = Math.log10(max);
        double step = (logMax - logMin) / (numPoints - 1);
        
        for (int i = 0; i < numPoints; i++) {
            double logValue = logMin + (i * step);
            int value = (int) Math.round(Math.pow(10, logValue));
            points.add(value);
        }
        
        return points;
    }
    
    private int calculateEmbeddingDim(int uniqueValues) {
        // Heuristic: embedding dim = min(sqrt(unique_values), 64)
        // This gives reasonable dimensions without exploding parameters
        return Math.min((int) Math.sqrt(uniqueValues), 64);
    }
    
    private int calculateHiddenSize(int scale) {
        // Heuristic: hidden size scales with log of unique values
        // Min 16, max 256 to keep network reasonable
        int size = (int) (16 * Math.log10(scale));
        return Math.max(16, Math.min(256, size));
    }
    
    private TestResult testAtScale(int uniqueValues, int embeddingDim, int hiddenSize, float learningRate) {
        // Simplified feature space - just 2 features to focus on scaling
        Feature[] features = {
            Feature.embedding(4, Math.min(embeddingDim, 4), "OS"),
            Feature.embedding(uniqueValues, embeddingDim, "SEGMENT")
        };
        
        Optimizer optimizer = new AdamWOptimizer(learningRate, 0f);

        NeuralNet net = NeuralNet.newBuilder()
                .setDefaultOptimizer(optimizer)
                .withGlobalGradientClipping(0f) // Re-enable clipping
                .layer(Layers.inputMixed(features))
                .layer(Layers.layerNorm())
                .layer(Layers.hiddenDenseRelu(hiddenSize))
                .layer(Layers.hiddenDenseRelu(hiddenSize / 2))
                .output(Layers.outputLinearRegression(1));
        
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Define pattern: 20% of segments are premium (high value)
        int numPremium = Math.max(1, uniqueValues / 5);
        Set<Integer> premiumSegments = new HashSet<>();
        Random rand = new Random(42);
        while (premiumSegments.size() < numPremium) {
            premiumSegments.add(rand.nextInt(uniqueValues));
        }
        
        // Training data
        List<Map<String, Object>> trainInputs = new ArrayList<>();
        List<Float> trainTargets = new ArrayList<>();
        
        // Generate training samples
        for (int i = 0; i < TRAINING_STEPS; i++) {
            int os = rand.nextInt(4);
            int segment = rand.nextInt(uniqueValues);
            
            Map<String, Object> input = Map.of(
                "OS", "os_" + os,
                "SEGMENT", "seg_" + segment
            );
            
            // Premium segments = $1.0, regular = $0.1
            float target = premiumSegments.contains(segment) ? 1.0f : 0.1f;
            
            trainInputs.add(input);
            trainTargets.add(target);
        }
        
        // Train in batches
        int batchSize = BATCH_SIZE; // Math.min(100, TRAINING_STEPS / 50);
        //int batchSize = Math.min(200, TRAINING_STEPS / 50);
        List<Float> penalties = new ArrayList<>();
        for (int i = 0; i < batchSize; i++) {
            penalties.add(-0.001f);
        }

        for (int i = 0; i < trainInputs.size(); i += batchSize) {
            int end = Math.min(i + batchSize, trainInputs.size());
            List<Map<String, Object>> batch = trainInputs.subList(i, end);
            List<Float> targets = trainTargets.subList(i, end);
            //model.train(batch.getFirst(), 0.0001f);
            //model.train(batch.getFirst(), targets.getFirst());
            //model.trainBatchMaps(batch, penalties);
            model.trainBatchMaps(batch, targets);
        }
        
        // Evaluate: test a sample of premium and regular segments
        float premiumSum = 0;
        int premiumCount = 0;
        float regularSum = 0;
        int regularCount = 0;
        
        // Test up to 100 segments to get average predictions
        int testCount = Math.min(100, uniqueValues);
        for (int i = 0; i < testCount; i++) {
            float pred = model.predictFloat(Map.of(
                "OS", "os_0",
                "SEGMENT", "seg_" + i
            ));
            
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
        float differentiation = premiumAvg - regularAvg;
        
        return new TestResult(
            uniqueValues, 
            learningRate,
            premiumAvg, 
            regularAvg, 
            differentiation,
            differentiation > SUCCESS_THRESHOLD
        );
    }
    
    private void printSummary(List<Integer> scalePoints, Map<String, List<TestResult>> results) {
        System.out.println("\n\n=== SUMMARY: Learning Success by Scale ===");
        System.out.println("(✓ = successful learning, ✗ = failed to learn)\n");
        
        // Header
        System.out.print("Scale     ");
        for (float lr : LEARNING_RATES) {
            System.out.printf("LR=%.4f  ", lr);
        }
        System.out.println();
        System.out.println("-".repeat(50));
        
        // Results grid
        for (int i = 0; i < scalePoints.size(); i++) {
            System.out.printf("%-10d", scalePoints.get(i));
            
            for (float lr : LEARNING_RATES) {
                String key = String.format("LR=%.4f", lr);
                TestResult result = results.get(key).get(i);
                System.out.printf("%-10s", result.success ? "✓" : "✗");
            }
            System.out.println();
        }
        
        // Find breaking points
        System.out.println("\n=== Breaking Points ===");
        for (float lr : LEARNING_RATES) {
            String key = String.format("LR=%.4f", lr);
            List<TestResult> lrResults = results.get(key);
            
            int lastSuccess = -1;
            for (int i = 0; i < lrResults.size(); i++) {
                if (lrResults.get(i).success) {
                    lastSuccess = i;
                }
            }
            
            if (lastSuccess >= 0 && lastSuccess < lrResults.size() - 1) {
                System.out.printf("LR=%.4f: Learning breaks between %d and %d unique values\n",
                    lr, scalePoints.get(lastSuccess), scalePoints.get(lastSuccess + 1));
            } else if (lastSuccess == lrResults.size() - 1) {
                System.out.printf("LR=%.4f: Still learning at maximum scale (%d)\n", 
                    lr, scalePoints.get(lastSuccess));
            } else {
                System.out.printf("LR=%.4f: Never achieved successful learning\n", lr);
            }
        }
        
        // Print differentiation values for best performing LR
        System.out.println("\n=== Differentiation Values (Premium - Regular) ===");
        System.out.println("Scale     Best LR    Differentiation");
        System.out.println("-".repeat(40));
        
        for (int i = 0; i < scalePoints.size(); i++) {
            float bestDiff = -1;
            float bestLR = 0;
            
            for (float lr : LEARNING_RATES) {
                String key = String.format("LR=%.4f", lr);
                TestResult result = results.get(key).get(i);
                if (result.differentiation > bestDiff) {
                    bestDiff = result.differentiation;
                    bestLR = lr;
                }
            }
            
            System.out.printf("%-10d%.4f     $%.3f\n", scalePoints.get(i), bestLR, bestDiff);
        }
    }
    
    private static class TestResult {
        final int scale;
        final float learningRate;
        final float premiumAvg;
        final float regularAvg;
        final float differentiation;
        final boolean success;
        
        TestResult(int scale, float learningRate, float premiumAvg, float regularAvg, 
                   float differentiation, boolean success) {
            this.scale = scale;
            this.learningRate = learningRate;
            this.premiumAvg = premiumAvg;
            this.regularAvg = regularAvg;
            this.differentiation = differentiation;
            this.success = success;
        }
    }
}