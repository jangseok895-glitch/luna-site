export async function onRequest() {
  return Response.json({
    ok: true,
    message: "Cloudflare Pages API 연결 성공"
  });
}