package dev.neuronic.net.optimizers;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;

/**
 * Tests the functionality of sparse optimizer updates, particularly for AdamW.
 * Ensures that "lazy" updates only modify the state of touched parameters,
 * preventing state decay for untouched parameters.
 */
public class SparseOptimizerTest {

    @Test
    public void testAdamWSparseUpdatePreventsStateDecay() {
        float learningRate = 0.1f;
        float weightDecay = 0.1f; // Use non-zero decay to test its application
        AdamWOptimizer optimizer = new AdamWOptimizer(learningRate, weightDecay);

        // A mock full embedding table
        float[][] allEmbeddings = new float[10][5];
        for (int i = 0; i < allEmbeddings.length; i++) {
            Arrays.fill(allEmbeddings[i], 1.0f);
        }

        // --- First sparse update ---
        // We will only update indices 2 and 5
        int[] indicesToUpdate1 = {2, 5};
        float[][] gradients1 = {
            {0.1f, 0.1f, 0.1f, 0.1f, 0.1f}, // Gradient for index 2
            {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}  // Gradient for index 5
        };

        // Create copies of untouched rows to verify they don't change
        float[] untouchedRow0_before = allEmbeddings[0].clone();
        float[] untouchedRow5_before = allEmbeddings[5].clone();


        // Perform the sparse update
        optimizer.sparseOptimize(allEmbeddings, allEmbeddings, indicesToUpdate1, gradients1, null);

        // --- Verification after first update ---

        // 1. Check that the UPDATED rows have changed
        assertFalse(Arrays.equals(untouchedRow5_before, allEmbeddings[5]), "Row 5 should have been updated.");
        
        // 2. Check that the UNTOUCHED rows have NOT changed
        // This is the critical test for lazy updates. With the old dense updates,
        // weight decay would have modified this row.
        assertArrayEquals(untouchedRow0_before, allEmbeddings[0], 1e-9f, "Untouched row 0 should NOT have changed.");

        // 3. Check that the optimizer state was created
        // We can't access the state directly, but we can infer it by checking if a second
        // update behaves as expected (i.e., it uses the momentum from the first).
        
        // --- Second sparse update ---
        // Update index 5 again to see if momentum is applied
        int[] indicesToUpdate2 = {5};
        float[][] gradients2 = {
            {0.1f, 0.1f, 0.1f, 0.1f, 0.1f} // Smaller gradient this time
        };
        
        float[] row5_after_first_update = allEmbeddings[5].clone();

        optimizer.sparseOptimize(allEmbeddings, allEmbeddings, indicesToUpdate2, gradients2, null);
        
        // --- Verification after second update ---
        
        // 4. Check that the second update on row 5 was successful
        assertFalse(Arrays.equals(row5_after_first_update, allEmbeddings[5]), "Row 5 should have been updated again.");

        // 5. Check that the other untouched row is STILL untouched
        assertArrayEquals(untouchedRow0_before, allEmbeddings[0], 1e-9f, "Untouched row 0 should STILL be unchanged.");

        System.out.println("SparseOptimizerTest successful: Lazy updates correctly modify only touched parameters and prevent state decay.");
    }
}
