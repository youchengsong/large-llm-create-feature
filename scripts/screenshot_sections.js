const puppeteer = require('puppeteer-core');

async function main() {
  const browser = await puppeteer.launch({
    executablePath: '/usr/bin/google-chrome',
    headless: 'shell',
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'],
    protocolTimeout: 180000,
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1100, height: 800, deviceScaleFactor: 2 });

  const url = 'file:///home/ubuntu/Projects/大模型提取特征/outputs/system_architecture_v4.html';
  await page.goto(url, { waitUntil: 'load', timeout: 60000 });
  await new Promise(r => setTimeout(r, 2000));

  const sections = [
    '#sec1', '#sec2', '#sec3', '#sec4', '#sec5',
    '#sec6', '#sec7', '#sec8', '#sec9', '#sec10',
    '#sec-example'
  ];

  const names = [
    'sec1_core_idea', 'sec2_architecture', 'sec3_knowledge_injection',
    'sec4_toolbox', 'sec5_verification', 'sec6_search_strategy',
    'sec7_memory', 'sec8_convergence', 'sec9_metrics',
    'sec10_comparison', 'sec_example'
  ];

  for (let i = 0; i < sections.length; i++) {
    try {
      const el = await page.$(sections[i]);
      if (el) {
        const box = await el.boundingBox();
        if (box) {
          await page.screenshot({
            path: `/tmp/docx_screenshots/${names[i]}.png`,
            clip: { x: box.x, y: box.y, width: box.width, height: Math.min(box.height, 4000) },
            type: 'png'
          });
          console.log(`OK: ${names[i]} (${Math.round(box.height)}px)`);
        }
      } else {
        console.log(`MISS: ${sections[i]}`);
      }
    } catch (e) {
      console.log(`ERR ${names[i]}: ${e.message.substring(0, 80)}`);
    }
  }

  // Architecture diagram specifically
  try {
    const arch = await page.$('.arch-diagram');
    if (arch) {
      const box = await arch.boundingBox();
      if (box) {
        await page.screenshot({
          path: '/tmp/docx_screenshots/arch_diagram.png',
          clip: { x: box.x, y: box.y, width: box.width, height: Math.min(box.height, 4000) },
          type: 'png'
        });
        console.log(`OK: arch_diagram (${Math.round(box.height)}px)`);
      }
    }
  } catch (e) {
    console.log(`ERR arch_diagram: ${e.message.substring(0, 80)}`);
  }

  // Flow containers
  try {
    const flows = await page.$$('.flow-container');
    for (let i = 0; i < flows.length; i++) {
      const box = await flows[i].boundingBox();
      if (box) {
        await page.screenshot({
          path: `/tmp/docx_screenshots/flow_${i}.png`,
          clip: { x: box.x, y: box.y, width: box.width, height: Math.min(box.height, 2000) },
          type: 'png'
        });
        console.log(`OK: flow_${i} (${Math.round(box.height)}px)`);
      }
    }
  } catch (e) {
    console.log(`ERR flow: ${e.message.substring(0, 80)}`);
  }

  await browser.close();
  console.log('All done');
}

main().catch(e => { console.error(e.message); process.exit(1); });
