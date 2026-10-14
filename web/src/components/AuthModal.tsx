"use client";

import React, { useState } from "react";
import { X, Lock, Mail, ShieldCheck, KeyRound } from "lucide-react";
import { UserIdentity } from "@/lib/types";

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  onLoginSuccess: (user: UserIdentity) => void;
}

export function AuthModal({ isOpen, onClose, onLoginSuccess }: AuthModalProps) {
  const [email, setEmail] = useState("admin@jewellery.internal");
  const [password, setPassword] = useState("dev_password_2026");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleFillDemo = () => {
    setEmail("admin@jewellery.internal");
    setPassword("dev_password_2026");
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });

      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data.error || "Authentication failed");
      }

      onLoginSuccess(data.user);
      onClose();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="bg-white rounded-2xl max-w-md w-full shadow-2xl border border-charcoal-200 overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        <div className="px-6 py-5 bg-ivory-100 border-b border-charcoal-100 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="p-2 bg-gold/15 rounded-lg text-gold-deep">
              <KeyRound className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-charcoal-900">Tenant Authentication</h2>
              <p className="text-xs text-charcoal-600">Enterprise RBAC Session</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-charcoal-400 hover:text-charcoal-700 hover:bg-charcoal-100 transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-xs text-red-700">
              {error}
            </div>
          )}

          <div>
            <label className="block text-xs font-medium text-charcoal-700 mb-1">
              Operator Email
            </label>
            <div className="relative">
              <Mail className="w-4 h-4 text-charcoal-400 absolute left-3 top-2.5" />
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full pl-9 pr-3 py-2 text-sm border border-charcoal-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-gold focus:border-transparent bg-ivory-50 text-charcoal-900"
                placeholder="admin@jewellery.internal"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-charcoal-700 mb-1">
              Password
            </label>
            <div className="relative">
              <Lock className="w-4 h-4 text-charcoal-400 absolute left-3 top-2.5" />
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full pl-9 pr-3 py-2 text-sm border border-charcoal-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-gold focus:border-transparent bg-ivory-50 text-charcoal-900"
                placeholder="••••••••••••"
              />
            </div>
          </div>

          <div className="bg-ivory-200/70 p-3 rounded-lg border border-charcoal-100">
            <div className="flex items-center justify-between text-xs">
              <span className="text-charcoal-600 flex items-center">
                <ShieldCheck className="w-3.5 h-3.5 text-gold-deep mr-1" />
                Pre-configured Demo Credentials
              </span>
              <button
                type="button"
                onClick={handleFillDemo}
                className="text-gold-deep hover:text-gold-dark font-medium underline"
              >
                Auto-fill
              </button>
            </div>
            <p className="text-[11px] text-charcoal-500 mt-1 font-mono">
              admin@jewellery.internal / dev_password_2026
            </p>
          </div>

          <div className="pt-2 flex items-center justify-end space-x-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-xs font-medium text-charcoal-700 hover:bg-charcoal-100 rounded-lg transition"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isLoading}
              className="px-5 py-2 text-xs font-medium bg-charcoal-900 text-ivory rounded-lg hover:bg-charcoal-800 transition disabled:opacity-50 shadow-sm"
            >
              {isLoading ? "Signing in..." : "Sign In"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
