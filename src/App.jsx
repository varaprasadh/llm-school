import { Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";
import ChapterPage from "./components/ChapterPage";
import Home from "./pages/Home";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="chapter/:slug" element={<ChapterPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
