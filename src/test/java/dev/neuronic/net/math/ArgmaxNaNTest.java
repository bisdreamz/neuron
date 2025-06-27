package dev.neuronic.net.math;

import dev.neuronic.net.common.Utils;

public class ArgmaxNaNTest {
    public static void main(String[] args) {
        float[] nanArray = {Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN};
        int result = Utils.argmax(nanArray);
        System.out.println("Argmax of NaN array: " + result);
        
        float[] mixedArray = {Float.NaN, 0.5f, Float.NaN, 0.3f, Float.NaN};
        int result2 = Utils.argmax(mixedArray);
        System.out.println("Argmax of mixed array: " + result2);
    }
}