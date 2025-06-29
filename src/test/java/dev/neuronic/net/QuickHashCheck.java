package dev.neuronic.net;

import dev.neuronic.net.layers.HashUtils;

public class QuickHashCheck {
    public static void main(String[] args) {
        // Check hash values for common strings
        String[] testStrings = {
            "test", "test_item", "domain_1", "app_1", "user_123",
            "example.com", "mobile_app", "desktop", "campaign_456"
        };
        
        System.out.println("Hash values for common strings:");
        for (String s : testStrings) {
            int hash = HashUtils.hashString(s);
            System.out.printf("%-20s: %d\n", s, hash);
        }
        
        // Check if any common string hashes to 0
        System.out.println("\nChecking 10000 strings for hash=0:");
        int zeroCount = 0;
        for (int i = 0; i < 10000; i++) {
            String s = "test_" + i;
            if (HashUtils.hashString(s) == 0) {
                System.out.println("Found hash=0 for: " + s);
                zeroCount++;
            }
        }
        System.out.println("Total strings with hash=0: " + zeroCount + " out of 10000");
    }
}