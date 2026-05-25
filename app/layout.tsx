import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Rubik's Cube RL Solver",
  description: "Minimal RL solver interface for an animated Rubik's Cube"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
