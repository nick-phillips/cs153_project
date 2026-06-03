import React from 'react';
import { createRoot } from 'react-dom/client';
import { HashRouter, Routes, Route } from 'react-router-dom';
import IndexPage from './routes/IndexPage';
import CompoundPage from './routes/CompoundPage';
import './styles.css';

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <HashRouter>
      <Routes>
        <Route path="/" element={<IndexPage />} />
        <Route path="/c/:id" element={<CompoundPage />} />
      </Routes>
    </HashRouter>
  </React.StrictMode>,
);
