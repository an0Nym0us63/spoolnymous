import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { useAuth } from "./store/auth";
import Layout from "./components/layout/Layout";
import Login from "./pages/Login";
import Home from "./pages/Home";
import Settings from "./pages/Settings";

function PrivateRoute({ children }) {
  const { isAuthenticated } = useAuth();
  return isAuthenticated ? children : <Navigate to="/login" replace />;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/" element={
        <PrivateRoute>
          <Layout />
        </PrivateRoute>
      }>
        <Route index element={<Home />} />
        <Route path="settings" element={<Settings />} />
      </Route>
    </Routes>
  );
}
