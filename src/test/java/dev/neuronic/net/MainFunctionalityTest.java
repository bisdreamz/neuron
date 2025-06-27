package dev.neuronic.net;

import dev.neuronic.net.optimizers.SgdOptimizer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test that main functionality still works after executor changes.
 */
class MainFunctionalityTest {

    @Test
    void testBasicNetworkTraining() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
                .input(4)
                .layer(Layers.hiddenDenseRelu(8, optimizer))
                .output(Layers.outputSoftmaxCrossEntropy(2, optimizer));
        
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] targets = {1.0f, 0.0f};
        
        // Get initial prediction
        float[] initialPred = net.predict(input);
        assertEquals(2, initialPred.length);
        
        // Train for a few iterations
        for (int i = 0; i < 10; i++) {
            net.train(input, targets);
        }
        
        // Get prediction after training
        float[] trainedPred = net.predict(input);
        assertEquals(2, trainedPred.length);
        
        // Verify it's a valid probability distribution
        float sum = trainedPred[0] + trainedPred[1];
        assertEquals(1.0f, sum, 0.01f, "Should be valid probability distribution");
        
        // Should have moved toward target (target[0] = 1.0)
        assertTrue(trainedPred[0] >= initialPred[0] - 0.1f, 
                  "Should have moved toward target or stayed stable");
    }
    
    @Test
    void testNetworkWithExecutorTraining() {
        java.util.concurrent.ExecutorService executor = 
            java.util.concurrent.Executors.newFixedThreadPool(2);
        
        try {
            SgdOptimizer optimizer = new SgdOptimizer(0.01f);
            
            NeuralNet net = NeuralNet.newBuilder()
                    .input(4)
                    .layer(Layers.hiddenDenseRelu(8, optimizer))
                    .executor(executor)  // Use executor
                    .output(Layers.outputSoftmaxCrossEntropy(2, optimizer));
            
            float[] input = {1.0f, 2.0f, 3.0f, 4.0f};
            float[] targets = {1.0f, 0.0f};
            
            // Get initial prediction
            float[] initialPred = net.predict(input);
            assertEquals(2, initialPred.length);
            
            // Train for a few iterations
            for (int i = 0; i < 10; i++) {
                net.train(input, targets);
            }
            
            // Get prediction after training
            float[] trainedPred = net.predict(input);
            assertEquals(2, trainedPred.length);
            
            // Verify it's a valid probability distribution
            float sum = trainedPred[0] + trainedPred[1];
            assertEquals(1.0f, sum, 0.01f, "Should be valid probability distribution");
            
        } finally {
            executor.shutdown();
        }
    }
}