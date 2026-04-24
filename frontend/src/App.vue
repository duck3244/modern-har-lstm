<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
</script>

<template>
  <div class="min-h-full flex flex-col">
    <header class="bg-white border-b border-slate-200 shadow-sm">
      <div class="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <h1 class="text-xl font-semibold text-slate-800">
          HAR LSTM Dashboard
        </h1>
        <nav class="flex gap-4">
          <RouterLink
            to="/dashboard"
            class="px-3 py-2 rounded-md text-sm font-medium text-slate-600 hover:bg-slate-100"
            active-class="bg-indigo-50 text-indigo-700"
          >
            Dashboard
          </RouterLink>
          <RouterLink
            to="/predict"
            class="px-3 py-2 rounded-md text-sm font-medium text-slate-600 hover:bg-slate-100"
            active-class="bg-indigo-50 text-indigo-700"
          >
            Predict
          </RouterLink>
        </nav>
      </div>
    </header>
    <main class="flex-1 max-w-7xl w-full mx-auto px-6 py-8">
      <RouterView />
    </main>
  </div>
</template>
