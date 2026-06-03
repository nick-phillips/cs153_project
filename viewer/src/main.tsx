import React from 'react';
import { createRoot } from 'react-dom/client';
import { HashRouter, Routes, Route } from 'react-router-dom';
import AppLayout from './routes/AppLayout';
import EmptyState from './routes/EmptyState';
import CompoundPage from './routes/CompoundPage';
import './styles.css';

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <HashRouter>
      <Routes>
        <Route path="/" element={<AppLayout />}>
          <Route index element={<EmptyState />} />
          <Route path="c/:id" element={<CompoundPage />} />
        </Route>
      </Routes>
    </HashRouter>
  </React.StrictMode>,
);
