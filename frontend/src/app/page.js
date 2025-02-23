import Image from "next/image";
import './App.css';
import QuestionForm from './QuestionForm';

export default function Home() {
  return (
    <div className="App">
      <header className="App-header">
        <Image src="logo.svg" width={100} height={100} className="App-logo" alt="logo" />
        <p>
          Web URLs RAG!
        </p>
        <QuestionForm />

      </header>
    </div>
  );
}
