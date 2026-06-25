import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",          // static HTML/CSS/JS for S3 + CloudFront
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;