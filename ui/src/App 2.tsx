/** App.tsx — router + layout. Theme/font-size are applied to <html>. */
import { useEffect } from "react";
import { BrowserRouter, Route, Routes } from "react-router";
import { AppShell } from "./components/layout/AppShell";
import { useSettingsStore } from "./stores/settingsStore";
import ChatView from "./routes/chat";
import TicketsView from "./routes/tickets";
import ManualsView from "./routes/manuals";
import ManualDetailView from "./routes/manual-detail";
import SettingsView from "./routes/settings";

export default function App() {
  const theme = useSettingsStore((s) => s.theme);
  const fontSize = useSettingsStore((s) => s.fontSize);

  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle("light", theme === "light");
    root.dataset.fontSize = fontSize;
  }, [theme, fontSize]);

  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<ChatView />} />
          <Route path="tickets" element={<TicketsView />} />
          <Route path="manuals" element={<ManualsView />} />
          <Route path="manuals/:id" element={<ManualDetailView />} />
          <Route path="settings" element={<SettingsView />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
