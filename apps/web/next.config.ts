import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone", // Enable standalone output for optimized deployment through docker
};

export default nextConfig;
