// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyAZCZyPTZHvXKwGs_wZIHDohul_CzZtpxw",
  authDomain: "sharebite-c7684.firebaseapp.com",
  projectId: "sharebite-c7684",
  storageBucket: "sharebite-c7684.firebasestorage.app",
  messagingSenderId: "387904923897",
  appId: "1:387904923897:web:fbe9baecb2db914459da28",
  measurementId: "G-NJ7NJYQT6T"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);