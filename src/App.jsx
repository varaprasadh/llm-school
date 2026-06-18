import { Routes, Route, Navigate, useParams } from "react-router-dom";
import Layout from "./components/Layout";
import ChapterPage from "./components/ChapterPage";
import RequireAuth from "./components/RequireAuth";
import Home from "./pages/Home";
import { allChapters } from "./data/chapters";

const FREE_SLUG = allChapters[0].slug;

/** Chapter 1 is public; every other chapter requires sign-in. */
function ChapterRoute() {
  const { slug } = useParams();
  if (slug === FREE_SLUG) return <ChapterPage />;
  return (
    <RequireAuth>
      <ChapterPage />
    </RequireAuth>
  );
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="chapter/:slug" element={<ChapterRoute />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
