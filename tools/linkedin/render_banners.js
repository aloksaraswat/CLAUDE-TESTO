const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' });
  for (const t of [
    { f:'banner_tech.html', out:'../../outputs/linkedin/banner_tech.png' },
    { f:'banner_backlit.html', out:'../../outputs/linkedin/banner_backlit.png' },
  ]) {
    const p = await b.newPage({ viewport:{width:1584,height:396}, deviceScaleFactor:2 });
    await p.goto('file://' + path.join(__dirname, t.f));
    await p.waitForTimeout(300);
    await (await p.$('.banner')).screenshot({ path: path.join(__dirname, t.out) });
    await p.close();
    console.log('saved', t.out);
  }
  await b.close();
})();
