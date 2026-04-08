const response = await fetch("http://localhost:8000/generate", {
  method: "POST",
  body: new URLSearchParams({
    task: "Write a LinkedIn post about AI automation",
    use_web_search: false,
  }),
});
const reader = response.body.getReader();
const decoder = new TextDecoder();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const lines = decoder.decode(value).split("\n");
  for (const line of lines) {
    if (line.startsWith("data: ")) {
      const { chunk } = JSON.parse(line.slice(6));
      process.stdout.write(chunk); // Stream to UI
    }
  }
}
