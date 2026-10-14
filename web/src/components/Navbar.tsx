"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Sparkles, Database, Activity, BarChart3, User, LogOut, CheckCircle2 } from "lucide-react";
import { AuthModal } from "./AuthModal";
import { UserIdentity } from "@/lib/types";

export function Navbar() {
  const pathname = usePathname();
  const [user, setUser] = useState<UserIdentity | null>(null);
  const [isAuthOpen, setIsAuthOpen] = useState(false);
  const [isOnline, setIsOnline] = useState(true);

  useEffect(() => {
    // Check session
    fetch("/api/auth/me")
      .then((res) => res.json())
      .then((data) => {
        if (data.authenticated && data.user) {
          setUser(data.user);
        }
      })
      .catch(() => {});
  }, []);

  const handleLogout = async () => {
    await fetch("/api/auth/logout", { method: "POST" });
    setUser(null);
  };

  const navItems = [
    { label: "Visual Search", href: "/", icon: Sparkles },
    { label: "Catalogue Browser", href: "/catalogue", icon: Database },
    { label: "Evaluation & Benchmarks", href: "/evaluation", icon: BarChart3 },
    { label: "System Status", href: "/status", icon: Activity },
  ];

  return (
    <>
      <header className="sticky top-0 z-40 bg-ivory-50/95 backdrop-blur-md border-b border-charcoal-200/60 shadow-subtle">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          {/* Logo & System Brand */}
          <div className="flex items-center space-x-6">
            <Link href="/" className="flex items-center space-x-3 group">
              <div className="w-9 h-9 rounded-xl bg-charcoal-900 text-gold flex items-center justify-center font-serif text-lg font-bold shadow-sm transition group-hover:bg-charcoal-800">
                ✦
              </div>
              <div className="flex flex-col">
                <span className="text-sm font-semibold tracking-wide text-charcoal-900 font-serif">
                  LUMEN <span className="font-sans text-xs text-charcoal-600 font-normal">| Jewellery Engine</span>
                </span>
                <span className="text-[10px] text-charcoal-500 font-mono tracking-tight">
                  MobileNetV4 1280-dim • Qdrant
                </span>
              </div>
            </Link>

            {/* Navigation links */}
            <nav className="hidden md:flex items-center space-x-1 pl-4 border-l border-charcoal-200">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = pathname === item.href;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                      isActive
                        ? "bg-charcoal-900 text-ivory shadow-sm"
                        : "text-charcoal-600 hover:text-charcoal-900 hover:bg-charcoal-100"
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5" />
                    <span>{item.label}</span>
                  </Link>
                );
              })}
            </nav>
          </div>

          {/* Right utility & Identity status */}
          <div className="flex items-center space-x-3">
            {/* Health pill */}
            <Link
              href="/status"
              className="hidden sm:flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium bg-emerald-50 text-emerald-800 border border-emerald-200 hover:bg-emerald-100 transition"
              title="System dependencies online"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              <span>Online</span>
            </Link>

            {/* Auth Session Button */}
            {user ? (
              <div className="flex items-center space-x-2 bg-ivory-200/80 px-3 py-1.5 rounded-lg border border-charcoal-200">
                <div className="w-5 h-5 rounded-full bg-gold-light text-gold-deep flex items-center justify-center text-xs font-bold">
                  {user.email ? user.email[0].toUpperCase() : "U"}
                </div>
                <div className="hidden lg:flex flex-col text-left">
                  <span className="text-xs font-medium text-charcoal-800 max-w-[130px] truncate">
                    {user.email || user.tenant_id}
                  </span>
                  <span className="text-[9px] text-charcoal-500 uppercase font-mono">
                    {user.tier || "ENTERPRISE"}
                  </span>
                </div>
                <button
                  onClick={handleLogout}
                  className="p-1 text-charcoal-400 hover:text-charcoal-700 hover:bg-charcoal-100 rounded transition ml-1"
                  title="Sign out"
                >
                  <LogOut className="w-3.5 h-3.5" />
                </button>
              </div>
            ) : (
              <button
                onClick={() => setIsAuthOpen(true)}
                className="flex items-center space-x-1.5 px-3 py-1.5 text-xs font-medium text-charcoal-700 bg-ivory-100 hover:bg-charcoal-100 border border-charcoal-200 rounded-lg transition"
              >
                <User className="w-3.5 h-3.5 text-charcoal-500" />
                <span>Operator Login</span>
              </button>
            )}
          </div>
        </div>

        {/* Mobile Navigation Row */}
        <div className="md:hidden flex items-center justify-around border-t border-charcoal-200/60 py-2 bg-ivory-100 px-2 overflow-x-auto">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center space-x-1 px-2.5 py-1 rounded-md text-[11px] font-medium whitespace-nowrap ${
                  isActive
                    ? "bg-charcoal-900 text-ivory"
                    : "text-charcoal-600 hover:bg-charcoal-200"
                }`}
              >
                <Icon className="w-3 h-3" />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </div>
      </header>

      <AuthModal
        isOpen={isAuthOpen}
        onClose={() => setIsAuthOpen(false)}
        onLoginSuccess={(u) => setUser(u)}
      />
    </>
  );
}
