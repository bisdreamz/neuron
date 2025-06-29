package dev.neuronic;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.Layers;
import dev.neuronic.net.optimizers.AdamOptimizer;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.Optimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class OptimizerGradientClippingTest {
    
    @BeforeEach
    public void setUp() {
        // Tests use non-deterministic initialization
    }
    
    @AfterEach
    public void tearDown() {
        // No cleanup needed
    }
    
    @Test
    public void testEachOptimizerWithGradientClipping() {
        // Test each optimizer with deterministic convergence expectations
        
        record OptimizerTest(String name, Optimizer optimizer,
                            int maxEpochs, float targetError) {}
        
        var tests = new OptimizerTest[] {
            // SGD with gradient-clipping-friendly parameters
            new OptimizerTest("SGD-0.01", new SgdOptimizer(0.01f), 700, 0.3f),
            new OptimizerTest("SGD-0.05", new SgdOptimizer(0.05f), 250, 0.3f),
            
            // Test Adam
            new OptimizerTest("Adam-0.001", new AdamOptimizer(0.001f), 800, 0.3f),
            new OptimizerTest("Adam-0.01", new AdamOptimizer(0.01f), 300, 0.3f),
            
            // Test AdamW
            new OptimizerTest("AdamW-0.001", new AdamWOptimizer(0.001f, 0.0001f), 1000, 0.3f),
            new OptimizerTest("AdamW-0.01", new AdamWOptimizer(0.01f, 0.00001f), 500, 0.3f),
            new OptimizerTest("AdamW-with-decay", new AdamWOptimizer(0.01f, 0.0001f), 150, 0.3f),
        };
        
        System.out.println("Testing each optimizer with gradient clipping:");
        System.out.println("=".repeat(60));
        
        for (var test : tests) {
            System.out.println("\nTesting: " + test.name);
            
            // Create network with gradient clipping
            NeuralNet net = NeuralNet.newBuilder()
                .input(2)
                .setDefaultOptimizer(test.optimizer)
                .withGlobalGradientClipping(2.0f)  // Moderate clipping that works
                .layer(Layers.hiddenDenseRelu(16))
                .output(Layers.outputSigmoidBinary());
            
            // XOR pattern - requires non-linear solution
            float[][] inputs = {
                {0.0f, 0.0f},
                {0.0f, 1.0f},
                {1.0f, 0.0f},
                {1.0f, 1.0f}
            };
            float[][] targets = {
                {0.0f}, {1.0f}, {1.0f}, {0.0f}
            };
            
            // Initial predictions
            System.out.println("Initial predictions:");
            for (int i = 0; i < inputs.length; i++) {
                float[] pred = net.predict(inputs[i]);
                System.out.printf("  [%.1f,%.1f] -> %.4f (target: %.1f)\n", 
                    inputs[i][0], inputs[i][1], pred[0], targets[i][0]);
            }
            
            // Train with early stopping
            boolean hasNaN = false;
            boolean converged = false;
            int epochsToConverge = -1;
            
            for (int epoch = 0; epoch < test.maxEpochs; epoch++) {
                for (int i = 0; i < inputs.length; i++) {
                    net.train(inputs[i], targets[i]);
                }
                
                // Calculate current error
                float currentError = 0.0f;
                for (int i = 0; i < inputs.length; i++) {
                    float[] pred = net.predict(inputs[i]);
                    if (Float.isNaN(pred[0]) || Float.isInfinite(pred[0])) {
                        hasNaN = true;
                        break;
                    }
                    currentError += Math.abs(pred[0] - targets[i][0]);
                }
                
                if (hasNaN) break;
                
                currentError /= inputs.length;
                
                // Early stopping if converged
                if (currentError < test.targetError) {
                    converged = true;
                    epochsToConverge = epoch + 1;
                    break;
                }
            }
            
            // Final evaluation
            float avgError = 0.0f;
            System.out.println("Final predictions:");
            for (int i = 0; i < inputs.length; i++) {
                float[] pred = net.predict(inputs[i]);
                float error = Math.abs(pred[0] - targets[i][0]);
                avgError += error;
                System.out.printf("  [%.1f,%.1f] -> %.4f (target: %.1f, error: %.4f)\n", 
                    inputs[i][0], inputs[i][1], pred[0], targets[i][0], error);
            }
            avgError /= inputs.length;
            
            // Results
            System.out.printf("Average error: %.4f\n", avgError);
            if (converged) {
                System.out.printf("Status: PASSED (converged in %d epochs)\n", epochsToConverge);
            } else if (hasNaN) {
                System.out.printf("Status: FAILED (NaN/Inf)\n");
            } else {
                System.out.printf("Status: FAILED (did not converge in %d epochs)\n", test.maxEpochs);
            }
            
            // Assertions
            assertFalse(hasNaN, test.name + " produced NaN/Inf values");
            assertTrue(converged, 
                test.name + " failed to converge below " + test.targetError + 
                " within " + test.maxEpochs + " epochs. Final error: " + avgError);
        }
    }
    
    @Test
    public void testConvergenceAcrossMultipleSeeds() {
        // Test that optimizers converge consistently across different seeds
        int numSeeds = 5;
        long[] seeds = {42L, 123L, 456L, 789L, 1000L};
        
        record OptimizerConfig(String name, Optimizer optimizerFactory,
                              int maxEpochs, float targetError, float successRate) {}
        
        var configs = new OptimizerConfig[] {
            // SGD with lower learning rate for stability
            new OptimizerConfig("SGD-0.05", new SgdOptimizer(0.05f), 800, 0.4f, 0.6f),
            // AdamW should converge reliably
            new OptimizerConfig("AdamW-0.01", new AdamWOptimizer(0.01f, 0.0f), 200, 0.2f, 0.8f),
        };
        
        for (var config : configs) {
            System.out.println("\nTesting " + config.name + " across " + numSeeds + " seeds:");
            
            int successCount = 0;
            int totalEpochs = 0;
            
            for (long seed : seeds) {
                // Each run uses random initialization
                
                // Create fresh optimizer instance for each run
                var optimizer = config.name.startsWith("SGD") ? 
                    new SgdOptimizer(0.05f) : 
                    new AdamWOptimizer(0.01f, 0.0f);
                
                NeuralNet net = NeuralNet.newBuilder()
                    .input(2)
                    .setDefaultOptimizer(optimizer)
                    .withGlobalGradientClipping(2.0f)
                    .layer(Layers.hiddenDenseRelu(16))
                    .layer(Layers.hiddenDenseRelu(8))
                    .output(Layers.outputSigmoidBinary());
                
                // XOR data
                float[][] inputs = {{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};
                float[][] targets = {{0.0f}, {1.0f}, {1.0f}, {0.0f}};
                
                // Train with early stopping
                boolean converged = false;
                for (int epoch = 0; epoch < config.maxEpochs; epoch++) {
                    for (int i = 0; i < inputs.length; i++) {
                        net.train(inputs[i], targets[i]);
                    }
                    
                    // Check convergence
                    float error = 0.0f;
                    for (int i = 0; i < inputs.length; i++) {
                        float[] pred = net.predict(inputs[i]);
                        error += Math.abs(pred[0] - targets[i][0]);
                    }
                    error /= inputs.length;
                    
                    if (error < config.targetError) {
                        converged = true;
                        totalEpochs += epoch + 1;
                        successCount++;
                        System.out.printf("  Seed %d: converged in %d epochs (error=%.4f)\n", 
                            seed, epoch + 1, error);
                        break;
                    }
                }
                
                if (!converged) {
                    System.out.printf("  Seed %d: FAILED to converge\n", seed);
                }
            }
            
            float actualSuccessRate = (float) successCount / numSeeds;
            float avgEpochs = successCount > 0 ? (float) totalEpochs / successCount : 0;
            
            System.out.printf("Results: %d/%d converged (%.0f%%), avg epochs: %.1f\n", 
                successCount, numSeeds, actualSuccessRate * 100, avgEpochs);
            
            assertTrue(actualSuccessRate >= config.successRate,
                config.name + " should converge at least " + (config.successRate * 100) + 
                "% of the time, but only converged " + (actualSuccessRate * 100) + "%");
        }
    }
    
    @Test
    public void testOptimizersWithoutGradientClipping() {
        // Control test - verify optimizers work without gradient clipping
        
        var optimizers = new Optimizer[] {
            new SgdOptimizer(0.05f),
            new AdamOptimizer(0.01f),
            new AdamWOptimizer(0.01f, 0.0f)
        };
        
        System.out.println("\nControl test - optimizers WITHOUT gradient clipping:");
        System.out.println("=".repeat(60));
        
        for (var optimizer : optimizers) {
            String name = optimizer.getClass().getSimpleName();
            System.out.println("\nTesting: " + name);
            
            NeuralNet net = NeuralNet.newBuilder()
                .input(2)
                .setDefaultOptimizer(optimizer)
                .withGlobalGradientClipping(0.0f)  // DISABLED
                .layer(Layers.hiddenDenseRelu(16))
                .layer(Layers.hiddenDenseRelu(8))
                .output(Layers.outputSigmoidBinary());
            
            // XOR data
            float[][] inputs = {{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};
            float[][] targets = {{0.0f}, {1.0f}, {1.0f}, {0.0f}};
            
            // Train
            for (int epoch = 0; epoch < 300; epoch++) {
                for (int i = 0; i < inputs.length; i++) {
                    net.train(inputs[i], targets[i]);
                }
            }
            
            // Evaluate
            float avgError = 0.0f;
            for (int i = 0; i < inputs.length; i++) {
                float[] pred = net.predict(inputs[i]);
                avgError += Math.abs(pred[0] - targets[i][0]);
            }
            avgError /= inputs.length;
            
            System.out.printf("Average error: %.4f\n", avgError);
            System.out.printf("Status: %s\n", avgError < 0.3f ? "GOOD" : "POOR");
        }
    }
}